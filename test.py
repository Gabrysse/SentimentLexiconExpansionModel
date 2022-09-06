import argparse
import torch

from main import unsupervised_review_sentiment
from neural.net_softmax import NetSoftmax
from dataset.Utilities import read_glove, getIMDBDF, getHotelReviewDF, getFakeNewsDF, getCoronaDF, getSpamDF


def word_ranking(word_dict, N=10, diff=False):
    top_neg_words = dict(sorted(word_dict.items(), key=lambda item: item[1])[0:N])
    top_pos_words = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True)[0:N])

    if diff:
        print(f"\n👍 TOP {N} WORDS CHANGED IN POSITIVE")
        for elem in list(top_pos_words.items()):
            print(elem)
        print(f"👎 TOP {N} WORDS CHANGED IN NEGATIVE")
        for elem in list(top_neg_words.items()):
            print(elem)
    else:
        print(f"👍 TOP {N} POSITIVE WORDS")
        for elem in list(top_pos_words.items()):
            print(elem)
        print(f"👎 TOP {N} NEGATIVE WORDS")
        for elem in list(top_neg_words.items()):
            print(elem)

    return top_pos_words, top_neg_words


def word_difference(dict1, dict2):
    difference = {}
    for key in dict2.keys():
        if key in dict1.keys():
            if (dict2[key][0] > 0 and dict1[key][0] < 0) or (dict2[key][0] < 0 and dict1[key][0] > 0):
                difference[key] = dict2[key][0] - dict1[key][0]

    return difference


def main(args):
    glove = read_glove()

    loaded_checkpoint = torch.load(args.checkpoint1)
    model1 = NetSoftmax(loaded_checkpoint['scale_min'], loaded_checkpoint['scale_max'])
    model1.load_state_dict(loaded_checkpoint['model_state_dict'])

    if args.checkpoint2 is not None:
        loaded_checkpoint = torch.load(args.checkpoint2)
        model2 = NetSoftmax(loaded_checkpoint['scale_min'], loaded_checkpoint['scale_max'])
        model2.load_state_dict(loaded_checkpoint['model_state_dict'])

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
            accuracy1, cache1 = unsupervised_review_sentiment(df, model1, glove)
            print(f"🏆 {name} accuracy {accuracy1}")
            if args.word_ranking:
                top_pos_words, top_neg_words = word_ranking(cache1)

            if args.checkpoint2 is not None:
                accuracy2, cache2 = unsupervised_review_sentiment(df, model2, glove)
                print(f"🏆 {name} accuracy {accuracy2}")
                if args.word_ranking:
                    top_pos_words, top_neg_words = word_ranking(cache2)
                    difference = word_difference(cache1, cache2)
                    word_ranking(difference, diff=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint1', type=str, default='net2.pth', help='Checkpoint path')
    parser.add_argument('--checkpoint2', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--unsup_dataset', type=str, help='Dataset used for unsupervised sentiment score. '
                                                          'Allowed values: imdb hotel fake_news covid_tweet spam')
    parser.add_argument('--word_ranking', action="store_true", help="Use this if you want to get the ranking of "
                                                                    "positive and negative word")
    args = parser.parse_args()
    main(args)
