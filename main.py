import argparse
import nltk
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from scipy import stats
from tqdm import tqdm
import time
import os

from dataset.PolarityDataset import PolarityDataset
from dataset.Utilities import read_vader, read_glove, dataPreparation, getAmazonDF, getIMDBDF, getHotelReviewDF, \
    getFakeNewsDF, getCoronaDF, getSpamDF
from neural.net_softmax import NetSoftmax
from neural.train import train
from preprocessing import seed_regression, seed_filter, tok, get_token_counts, train_linear_model, seed_filter2


def correlation_with_VADER(seed, vader, embeddings_index, net):
    polarities_vader = []
    polarities_seed = []
    polarities_net = []
    for token in vader.keys():
        polarities_vader.append(vader[token])

        try:
            polarities_seed.append(seed[token])
        except:
            polarities_seed.append(0)

        try:
            polarities_net.append(net(torch.tensor(embeddings_index[token]).unsqueeze(dim=0)).detach().item())
        except:
            polarities_net.append(0)

    polarities_vader = np.array(polarities_vader)
    polarities_seed = np.array(polarities_seed)
    polarities_net = np.array(polarities_net)

    # print(polarities_vader.shape)
    # print(polarities_seed.shape)
    # print(polarities_net.shape)

    print(f"Correlation SEED-VADER: {stats.pearsonr(polarities_vader, polarities_seed)[0]}")
    print(f"Correlation NETPREDICTION-VADER: {stats.pearsonr(polarities_vader, polarities_net)[0]}")


def unsupervised_review_sentiment(df, net, embeddings_index):
    with torch.no_grad():
        net.eval()
        print("⚙ Calculating accuracy...")
        accuracy = 0
        skipped = 0

        cache = dict()

        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            text = row['text']
            label = row['label']

            text_tok = tok(text)

            if len(text_tok) > 0:
                prediction = 0
                for word in text_tok:
                    try:
                        cached = cache.get(word)
                        if cached is None:
                            cached = net(torch.tensor(embeddings_index[word]).unsqueeze(dim=0)).detach().item()
                            cache.update({word: cached})
                        prediction += cached

                        # prediction += net(torch.tensor(embeddings_index[word]).unsqueeze(dim=0)).detach().item()
                    except:
                        pass

                prediction_score = prediction / len(text_tok)

                if (label == -1 and prediction_score < 0) or (label == 1 and prediction_score > 0):
                    accuracy += 1
            else:
                skipped += 1

        accuracy = round(accuracy / (len(df) - skipped), 6)

        return accuracy, cache


def domain_generic(vader, embeddings_index, ckp_name):
    # VALIDATION WITH VADER
    tokens, embeds, polarities, bucket = dataPreparation(vader, embeddings_index)

    train_tok, test_tok, train_emb, test_emb, train_pol, test_pol, train_bck, test_bck = train_test_split(tokens,
                                                                                                          embeds,
                                                                                                          polarities,
                                                                                                          bucket,
                                                                                                          test_size=0.2,
                                                                                                          stratify=bucket,
                                                                                                          shuffle=True)
    train_tok, val_tok, train_emb, val_emb, train_pol, val_pol = train_test_split(train_tok, train_emb, train_pol,
                                                                                  test_size=0.25, stratify=train_bck,
                                                                                  shuffle=True)

    scale_max = np.max(polarities)
    scale_min = np.min(polarities)

    glove_dataset = PolarityDataset(train_emb, train_pol)
    glove_dataset_eval = PolarityDataset(val_emb, val_pol)
    glove_dataset_test = PolarityDataset(test_emb, test_pol)

    train_dataloader = DataLoader(glove_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    eval_dataloader = DataLoader(glove_dataset_eval, batch_size=1, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(glove_dataset_test, batch_size=1, shuffle=True, num_workers=2)

    net1 = NetSoftmax(scale_min, scale_max)
    train(net1, train_dataloader, eval_dataloader)
    checkpoint = {
        'scale_max': scale_max,
        'scale_min': scale_min,
        'model_state_dict': net1.state_dict()
    }
    torch.save(checkpoint, f"net1_{ckp_name}.pth")

    return net1


def domain_specific(seed, vader, embeddings_index, ckp_name):
    tokens, embeds, polarities, _ = dataPreparation(seed, embeddings_index)

    train_tok, test_tok, train_emb, test_emb, train_pol, test_pol = train_test_split(tokens, embeds, polarities,
                                                                                     test_size=0.2, shuffle=True)
    train_tok, val_tok, train_emb, val_emb, train_pol, val_pol = train_test_split(train_tok, train_emb, train_pol,
                                                                                  test_size=0.25, shuffle=True)

    scale_max = np.max(polarities)
    scale_min = np.min(polarities)

    glove_dataset = PolarityDataset(train_emb, train_pol)
    glove_dataset_eval = PolarityDataset(val_emb, val_pol)
    glove_dataset_test = PolarityDataset(test_emb, test_pol)

    train_dataloader = DataLoader(glove_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    eval_dataloader = DataLoader(glove_dataset_eval, batch_size=1, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(glove_dataset_test, batch_size=1, shuffle=True, num_workers=2)

    net2 = NetSoftmax(scale_min, scale_max)
    train(net2, train_dataloader, eval_dataloader)
    checkpoint = {
        'scale_max': scale_max,
        'scale_min': scale_min,
        'model_state_dict': net2.state_dict()
    }
    torch.save(checkpoint, f"net2_{ckp_name}.pth")

    return net2


def main(args):
    nltk.download('punkt')

    vader = read_vader()
    glove = read_glove()

    if args.exp == "d_generic":
        print("\n **** DOMAIN GENERIC SENTIMENT SCORE ****\n")
        net1 = domain_generic(vader, glove, os.path.basename(args.dataset).split('.')[0])

        # TEST
        words = ["like", "love", "amazing", "excellent", "terrible", "awful", "ugly", "complaint"]

        net1.eval()

        for word in words:
            try:
                print("Predicted", word, net1(torch.tensor(glove[word]).unsqueeze(dim=0)).detach().item())
                print("Ground truth", word, vader[word])
            except:
                pass
            print("\n")
        ###################################################################################################
    elif args.exp == "d_specific":
        print("\n **** DOMAIN SPECIFIC SENTIMENT SCORE ****\n")

        df0 = getAmazonDF(args.dataset, args.filter_year)
        # vectorizer, regression = seed_regression(df0)
        # seed = seed_filter(df0, vectorizer, regression, frequency=500)

        X, features_list = get_token_counts(df0.reviewText)
        coeff = train_linear_model(X, df0.overall)
        seed = seed_filter2(X, features_list, coeff, frequency=500)
        print(f"Seed length: {len(seed)}")

        net2 = domain_specific(seed, vader, glove, f"{os.path.basename(args.dataset).split('.')[0]}_{args.filter_year}")
        correlation_with_VADER(seed, vader, glove, net2)
        ###################################################################################################
    elif args.exp == "unsup_sent":
        print("\n **** UNSUPERVISED REVIEW SENTIMENT CLASSIFICATION ****\n")

        df0 = getAmazonDF(args.dataset, args.filter_year)
        X, features_list = get_token_counts(df0.reviewText)
        coeff = train_linear_model(X, df0.overall)
        seed = seed_filter2(X, features_list, coeff, frequency=500)
        net2 = domain_specific(seed, vader, glove, f"{os.path.basename(args.dataset).split('.')[0]}_{args.filter_year}")

        for dataset in args.unsup_dataset.split(" "):
            df = None
            if dataset == "imdb":
                name = "IMDb"
                df = getIMDBDF()
            elif dataset == "hotel":
                name = "Hotel Review"
                df = getHotelReviewDF()
            elif dataset == "fake_news":
                name = "Fake news"
                df = getFakeNewsDF()
            elif dataset == "covid_tweet":
                name = "Corona virus tweet"
                df = getCoronaDF()
            elif dataset == "spam":
                name = "Spam email"
                df = getSpamDF()

            if df is not None:
                accuracy, cache = unsupervised_review_sentiment(df, net2, glove)
                print(f"🏆 {name} accuracy {accuracy}")

        ###################################################################################################ù
    elif args.exp == "fake_news":
        # 1. Leggere dataset delle fake news https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
        # 2. Tokenizzare
        # 3. Per ogni token poi ottenere l'embedding di glove
        # 4. Per ogni dominio usare NET2 per ottenere la predizione per quel token
        # 5. Confrontare i valori ottenuti per ogni dominio e cercare similarità
        pass
    elif args.exp == "cross_lingual":
        # Io lo inserirei così facciamo vedere che qualcosa abbiamo fatto.
        # Lo facciamo con i domini che abbiamo trovato e pace al signore.
        pass


if __name__ == '__main__':
    # params = [
    #     '--num_epochs', '100',
    #     '--learning_rate', '2.5e-2',
    #     '--data', '../datasets/CamVid/',
    #     '--num_workers', '8',
    #     '--batch_size', '4',
    #     '--optimizer', 'sgd',
    #     '--checkpoint_step', '2'
    # ]
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="d_specific", help='')
    parser.add_argument('--dataset', type=str, help='The amazon review dataset path you want to use')
    parser.add_argument('--filter_year', action='store_true', help='Consider only the review < July 2014')
    parser.add_argument('--unsup_dataset', type=str, help='Dataset used for unsupervised sentiment score. '
                                                          'Allowed values: imdb hotel fake_news covid_tweet spam')

    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=100, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')

    args = parser.parse_args()
    main(args)
