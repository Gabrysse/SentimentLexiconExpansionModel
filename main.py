import argparse
import nltk
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset.PolarityDataset import PolarityDataset
from dataset.Utilities import read_vader, read_glove, dataPreparation
from neural.net_softmax import NetSoftmax
from neural.train import train

nltk.download('punkt')


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=100, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
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

    args = parser.parse_args(params)

    # VALIDATION WITH VADER
    vader = read_vader()
    embeddings_index = read_glove()

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

    net = NetSoftmax(scale_min, scale_max)

    train(net, train_dataloader, eval_dataloader)

    # TEST
    words = ["like", "love", "amazing", "excellent", "terrible", "awful", "ugly", "complaint"]

    net.eval()

    for word in words:
        try:
            print("Predicted", word, net(torch.tensor(embeddings_index[word]).unsqueeze(dim=0)).detach().item())
            print("Ground truth", word, vader[word])
        except:
            pass
        print("\n")

    #######################


if __name__ == '__main__':
    params = [
        '--num_epochs', '100',
        '--learning_rate', '2.5e-2',
        '--data', '../datasets/CamVid/',
        '--num_workers', '8',
        '--batch_size', '4',
        '--optimizer', 'sgd',
        '--checkpoint_step', '2'
    ]
    main(params)
