import argparse
import nltk
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from scipy import stats
import os

from dataset.PolarityDataset import PolarityDataset
from dataset.Utilities import read_vader, read_glove, data_preparation, getAmazonDF
from neural.net_softmax import NetSoftmax
from neural.train import train
from preprocessing import get_token_counts, train_linear_model, seed_filter


def correlation_with_VADER(seed, vader, embeddings_index, net):
    """
    Function used to calculate the correlation with VADER.
        :param seed: Seed data dictonary
        :param vader: VADER lexicon dictonary
        :param embeddings_index: Embeddings dictonary (e.g. Glove)
        :param net: Neural network object
        :return: None
    """
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

    print(f"Correlation SEED-VADER: {stats.pearsonr(polarities_vader, polarities_seed)[0]}")
    print(f"Correlation NETPREDICTION-VADER: {stats.pearsonr(polarities_vader, polarities_net)[0]}")


def domain_generic(vader, embeddings_index):
    """
    This function perform the domain generic experiments. More information in the report.
        :param vader: VADER lexicon dictonary
        :param embeddings_index: Embeddings dictonary (e.g. Glove)
        :return: trained neural network
    """
    tokens, embeds, polarities, bucket = data_preparation(vader, embeddings_index)

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

    polarity_dataset = PolarityDataset(train_emb, train_pol)
    polarity_dataset_eval = PolarityDataset(val_emb, val_pol)
    polarity_dataset_test = PolarityDataset(test_emb, test_pol)

    train_dataloader = DataLoader(polarity_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    eval_dataloader = DataLoader(polarity_dataset_eval, batch_size=1, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(polarity_dataset_test, batch_size=1, shuffle=True, num_workers=2)

    net1 = NetSoftmax(scale_min, scale_max)
    train(net1, train_dataloader, eval_dataloader)
    checkpoint = {
        'scale_max': scale_max,
        'scale_min': scale_min,
        'model_state_dict': net1.state_dict()
    }
    torch.save(checkpoint, f"net1.pth")

    return net1


def domain_specific(seed, embeddings_index, ckp_name):
    """
    This function perform the domain specific experiments. More information in the report.
        :param seed: Seed data dictonary
        :param embeddings_index: Embeddings dictonary (e.g. Glove)
        :param ckp_name: Checkpoint name
        :return: trained neural network
    """

    tokens, embeds, polarities, _ = data_preparation(seed, embeddings_index)

    train_tok, test_tok, train_emb, test_emb, train_pol, test_pol = train_test_split(tokens, embeds, polarities,
                                                                                     test_size=0.2, shuffle=True)
    train_tok, val_tok, train_emb, val_emb, train_pol, val_pol = train_test_split(train_tok, train_emb, train_pol,
                                                                                  test_size=0.25, shuffle=True)

    scale_max = np.max(polarities)
    scale_min = np.min(polarities)

    polarity_dataset = PolarityDataset(train_emb, train_pol)
    polarity_dataset_eval = PolarityDataset(val_emb, val_pol)
    polarity_dataset_test = PolarityDataset(test_emb, test_pol)

    train_dataloader = DataLoader(polarity_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    eval_dataloader = DataLoader(polarity_dataset_eval, batch_size=1, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(polarity_dataset_test, batch_size=1, shuffle=True, num_workers=2)

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
        net1 = domain_generic(vader, glove)

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
    elif args.exp == "d_specific":
        print("\n **** DOMAIN SPECIFIC SENTIMENT SCORE ****\n")

        df0 = getAmazonDF(args.dataset, args.filter_year)
        X, features_list = get_token_counts(df0.reviewText)
        coeff = train_linear_model(X, df0.overall)
        seed = seed_filter(X, features_list, coeff, frequency=500)
        print(f"Seed length: {len(seed)}")

        net2 = domain_specific(seed, glove, f"{os.path.basename(args.dataset).split('.')[0]}_{args.filter_year}")
        correlation_with_VADER(seed, vader, glove, net2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="d_specific", help='')
    parser.add_argument('--dataset', type=str, help='The amazon review dataset path you want to use')
    parser.add_argument('--filter_year', action='store_true', help='Consider only the review < July 2014')
    parser.add_argument('--unsup_dataset', type=str, help='Dataset used for unsupervised sentiment score. '
                                                          'Allowed values: imdb hotel fake_news covid_tweet spam')

    args = parser.parse_args()
    main(args)
