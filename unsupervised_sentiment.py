import argparse
import torch
from tqdm import tqdm

from neural.net_softmax import NetSoftmax
from dataset.Utilities import read_glove, getIMDBDF, getHotelReviewDF, getFakeNewsDF, getCoronaDF, getSpamDF
from preprocessing import tok


def unsupervised_review_sentiment(df, net, embeddings_index):
    """
    Function used to calculate the accuracy of the unsupervised review sentiment experiment.
    More information in the report.
        :param df: Pandas dataframe that need to be processed. Must be in the form [Text,Label]
        :param net: trained neural network
        :param embeddings_index: embeddings dictionary containg {word: embeddings}
        :return: Accuracy obtained, cache vector containing the word occurred and their predicted score by the NN
    """
    with torch.no_grad():
        net.eval()
        print("\n⚙ Calculating accuracy...")
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
                            cached = (net(torch.tensor(embeddings_index[word]).unsqueeze(dim=0)).detach().item(), 1)
                            cache.update({word: cached})
                        else:
                            cached = (cached[0], cached[1] + 1)
                            cache.update({word: cached})
                        prediction += cached[0]
                    except:
                        pass

                prediction_score = prediction / len(text_tok)

                if (label == -1 and prediction_score < 0) or (label == 1 and prediction_score > 0):
                    accuracy += 1
            else:
                skipped += 1

        accuracy = round(accuracy / (len(df) - skipped), 6)

        return accuracy, cache


def word_ranking(word_dict, N=20, diff=False):
    """
    Function used to get the ranking of the top N positive and negative words
        :param word_dict: word dictionary containing {word: sentiment_score}
        :param N: number of element in the ranking
        :param diff: parameter used when word rankin is called after the method word_difference
        :return: None
    """
    top_neg_words = dict(sorted(word_dict.items(), key=lambda item: item[1][0])[0:N])
    top_pos_words = dict(sorted(word_dict.items(), key=lambda item: item[1][0], reverse=True)[0:N])

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


def word_difference(dict1, dict2):
    """
    Function used to calculate the difference between two word dictionary in order to get the word that change polarity
        :param dict1: word dictionary containing {word: sentiment_score}
        :param dict2: word dictionary containing {word: sentiment_score}
        :return: dictonary containing the difference between the polarity of dict2 and dict1
    """
    difference = {}
    for key in dict2.keys():
        if key in dict1.keys():
            if (dict2[key][0] > 0.05 and dict1[key][0] < -0.05) or (dict2[key][0] < -0.05 and dict1[key][0] > 0.05):
                difference[key] = (dict2[key][0] - dict1[key][0], dict2[key][1])

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
                word_ranking(cache1)

            if args.checkpoint2 is not None:
                accuracy2, cache2 = unsupervised_review_sentiment(df, model2, glove)
                print(f"🏆 {name} accuracy {accuracy2}")
                if args.word_ranking:
                    word_ranking(cache2)
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
