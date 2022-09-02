import argparse
import torch

from main import unsupervised_review_sentiment
from neural.net_softmax import NetSoftmax
from dataset.Utilities import read_glove, getIMDBDF, getHotelReviewDF, getFakeNewsDF, getCoronaDF, getSpamDF


def main(args):
    glove = read_glove()

    loaded_checkpoint = torch.load("net2.pth")
    model = NetSoftmax(loaded_checkpoint['scale_min'], loaded_checkpoint['scale_max'])
    model.load_state_dict(loaded_checkpoint['model_state_dict'])

    with torch.no_grad():
        model.eval()

        for dataset in args.unsup_dataset.split(" "):
            if dataset == "imdb":
                imdb = getIMDBDF()
                accuracy = unsupervised_review_sentiment(imdb, model, glove)
                print(f"IMDb accuracy {accuracy}")
            elif dataset == "hotel":
                hotel = getHotelReviewDF()
                accuracy = unsupervised_review_sentiment(hotel, model, glove)
                print(f"Hotel Review accuracy {accuracy}")
            elif dataset == "fake_news":
                fake_news = getFakeNewsDF()
                accuracy = unsupervised_review_sentiment(fake_news, model, glove)
                print(f"Fake news accuracy {accuracy}")
            elif dataset == "covid_tweet":
                covid_tweet = getCoronaDF()
                accuracy = unsupervised_review_sentiment(covid_tweet, model, glove)
                print(f"Corona virus tweet accuracy {accuracy}")
            elif dataset == "spam":
                spam = getSpamDF()
                accuracy = unsupervised_review_sentiment(spam, model, glove)
                print(f"Spam email accuracy {accuracy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unsup_dataset', type=str, help='Dataset used for unsupervised sentiment score. '
                                                          'Allowed values: imdb hotel fake_news covid_tweet spam')
    args = parser.parse_args()
    main(args)
