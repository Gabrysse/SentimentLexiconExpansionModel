import gzip
import json
import math
import numpy as np
import pandas as pd


def getAmazonDF(path, filter_year=True):
    """
    Function used to read the Amazon review dataset.
        :param path: path to the .gz file
        :param filter_year: True if you want to select the time period May 1996-July 2014.
        Otherwise, the time period is May 1996-October 2018
        :return: dataframe containing [ReviewText,Label]
    """
    print(f"📖 Reading review dataset [{path}, {filter_year}]...")
    review_dict = {}

    for i, line in enumerate(gzip.open(path, 'rb')):
        review = json.loads(line)
        if 'reviewText' in review and 'overall' in review:
            if filter_year:
                if review['unixReviewTime'] < 1406851200:
                    review_dict[i] = {'overall': review['overall'], 'reviewText': review['reviewText']}
            else:
                review_dict[i] = {'overall': review['overall'], 'reviewText': review['reviewText']}

    df = pd.DataFrame.from_dict(review_dict, orient='index')
    df = df[df["overall"] != 3]
    df['overall'] = df.apply(lambda row: (row["overall"] > 3) * 2 - 1, axis=1)
    df = df.fillna("")
    return df


def getIMDBDF(path="IMDB Dataset.csv"):
    """
    Function used to read the IMDb dataset.
        :param path: path to the .csv file
        :return: dataframe containing [Text,Label]
    """
    print(f"\n📖 Reading IMDb review dataset...")
    df = pd.read_csv(path)
    df.rename(columns={"review": "text", "sentiment": "label"}, inplace=True)
    df["label"] = df["label"].apply(lambda x: -1 if x == "negative" else 1)

    return df


def getHotelReviewDF(path="Hotel_Reviews.csv"):
    """
    Function used to read the Hotel review dataset.
        :param path: path to the .csv file
        :return: dataframe containing [Text,Label]
    """
    print(f"\n📖 Reading Hotel review dataset...")
    df = pd.read_csv(path)[["Negative_Review", "Positive_Review", "Reviewer_Score"]]
    df.loc[:, 'Positive_Review'] = df.Positive_Review.apply(lambda x: x.replace('No Positive', ''))
    df.loc[:, 'Negative_Review'] = df.Negative_Review.apply(lambda x: x.replace('No Negative', ''))
    df['text'] = df.Positive_Review + df.Negative_Review
    df["label"] = df["Reviewer_Score"].apply(lambda x: -1 if x < 7 else 1)

    return df[["text", "label"]]


def getCoronaDF(path="Corona_NLP_train.csv"):
    """
    Function used to read the Coronavirus tweet dataset.
        :param path: path to the .csv file
        :return: dataframe containing [Text,Label]
    """
    print(f"\n📖 Reading Corona tweet dataset...")
    df = pd.read_csv(path, encoding="ISO-8859-1")[["OriginalTweet", "Sentiment"]]

    df = df[df["Sentiment"] != "Neutral"]

    df['text'] = df.OriginalTweet
    df['label'] = df['Sentiment'].apply(lambda x: 1 if (x == "Positive" or x == "Extremely Positive") else -1 if (x == "Negative" or x == "Extremely Negative") else 0)

    df = df[['text', 'label']]

    return df


def getSpamDF(path="SPAM text message 20170820 - Data.csv"):
    """
    Function used to read the Spam messages dataset.
        :param path: path to the .csv file
        :return: dataframe containing [Text,Label]
    """
    print(f"\n📖 Reading SPAM text message dataset...")
    df = pd.read_csv(path)

    df['text'] = df.Message
    df['label'] = df['Category'].apply(lambda x: 1 if x == "ham" else -1 if x == "spam" else 0)

    df = df[['text', 'label']]

    return df


def getFakeNewsDF(true_news_path="True.csv", fake_news_path="Fake.csv"):
    """
    Function used to read the Fake news dataset.
        :param true_news_path: path to True.csv file
        :param fake_news_path: path to False.csv file
        :return: dataframe containing [Text,Label]
    """
    print(f"\n📖 Reading fake news dataset...")
    truedf = pd.read_csv(true_news_path)
    fakedf = pd.read_csv(fake_news_path)

    truedf.drop(columns=["title", "subject", "date"], inplace=True)
    fakedf.drop(columns=["title", "subject", "date"], inplace=True)
    truedf['label'] = 1
    fakedf['label'] = -1
    df = pd.concat([truedf, fakedf]).reset_index(drop=True)

    return df


def read_vader():
    """
    Function used to read the VADER lexicon
        :return: dictonary containg {word: sentiment_score}
    """
    print('\nIndexing VADER word vectors.')

    vader = {}
    f = open('vader_lexicon.txt', encoding='utf-8')
    for line in f:
        values = line.split('\t')
        word = values[0]
        coef = float(values[1])
        vader[word] = coef
    f.close()

    print('Found %s word vectors.' % len(vader))

    return vader


def read_glove():
    """
    Function used to read the GloVe CommonCrawl embeddings.
        :return: dictonary containg {word: embeddings}
    """
    print('Indexing GLOVE word vectors.')

    embeddings_index = {}
    f = open('glove.840B.300d.txt', encoding='utf-8')
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


def data_preparation(dataset, embeddings):
    """
    Function used to prepare data for training in the domain specific expertiment.
        :param dataset: dictionary containing {word: sentiment_score}
        :param embeddings: embeddigs dictionary containing {word: embeddings}
        :return: token list, embeddings list, polarities list, bucket list
    """
    tokens = []
    embeds = []
    polarities = []
    bucket = []

    for tok in dataset.keys():
        try:
            polarity = dataset[tok]
            embed = embeddings[tok]
            tokens.append(tok)
            embeds.append(np.array(embed))
            polarities.append(polarity)
            if polarity < 0:
                bucket.append(str(math.floor(polarity)))
            elif polarity > 0:
                bucket.append(str(math.ceil(polarity)))
            else:
                bucket.append("0")
        except:
            pass

    embeds = np.array(embeds)
    polarities = np.array(polarities)

    return tokens, embeds, polarities, bucket
