import argparse
import torch

from main import unsupervised_review_sentiment
from neural.net_softmax import NetSoftmax
from dataset.Utilities import read_glove, getIMDBDF, getHotelReviewDF, getFakeNewsDF, getCoronaDF, getSpamDF


def word_ranking(word_dict, N=10):
    top_neg_words = dict(sorted(word_dict.items(), key=lambda item: item[1])[0:N])
    top_pos_words = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True)[0:N])

    return top_pos_words, top_neg_words


def main(args):
    glove = read_glove()

    loaded_checkpoint = torch.load(args.checkpoint)
    model = NetSoftmax(loaded_checkpoint['scale_min'], loaded_checkpoint['scale_max'])
    model.load_state_dict(loaded_checkpoint['model_state_dict'])

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
            accuracy, cache = unsupervised_review_sentiment(df, model, glove)
            print(f"🏆 {name} accuracy {accuracy}")
            if args.word_ranking:
                top_pos_words, top_neg_words = word_ranking(cache)
                print("⬆ POSITIVE WORDS RANKING ⬆")
                for elem in list(top_pos_words.items()):
                    print(elem)
                print("⬇ NEGATIVE WORDS RANKING ⬇")
                for elem in list(top_neg_words.items()):
                    print(elem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='net2.pth', help='Checkpoint path')
    parser.add_argument('--unsup_dataset', type=str, help='Dataset used for unsupervised sentiment score. '
                                                          'Allowed values: imdb hotel fake_news covid_tweet spam')
    parser.add_argument('--word_ranking', action="store_true", help="Use this if you want to get the ranking of "
                                                                    "positive and negative word")
    args = parser.parse_args()
    main(args)
