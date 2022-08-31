import gzip
import json
import math
import numpy as np
import pandas as pd


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getAmazonDF(path, filter_year=True):
    print(f"Reading review dataset [{path}, {filter_year}]...")
    review_dict = {}
    # i = 0
    # for d in parse(path):
    #     review_dict[i] = d
    #     i += 1

    for i, line in enumerate(gzip.open(path, 'rb')):
        review = json.loads(line)
        if 'reviewText' in review and 'overall' in review:
            if filter_year:
                if review['unixReviewTime'] < 1406851200:
                    review_dict[i] = {'overall': review['overall'], 'reviewText': review['reviewText']}
            else:
                review_dict[i] = {'overall': review['overall'], 'reviewText': review['reviewText']}

    df = pd.DataFrame.from_dict(review_dict, orient='index')
    # df = df[df['unixReviewTime'] < 1406851200]  # remove the reviews
    # df = df.drop(
    #     columns=["verified", "reviewTime", "reviewerID", "asin", "reviewerName", "summary", "unixReviewTime", "vote",
    #              "style", "image"])
    df = df[df["overall"] != 3]
    df['overall'] = df.apply(lambda row: (row["overall"] > 3) * 2 - 1, axis=1)
    df = df.fillna("")
    return df


def getIMDBDF(path="IMDB Dataset.csv"):
    df = pd.read_csv(path)
    df.rename(columns={"review": "text", "sentiment": "label"}, inplace=True)
    df["label"] = df["label"].apply(lambda x: -1 if x < "negative" else 1)

    return df


def getHotelReviewDF(path):
    df = pd.read_csv("/kaggle/input/515k-hotel-reviews-data-in-europe/Hotel_Reviews.csv")[
        ["Negative_Review", "Positive_Review", "Reviewer_Score"]]
    df.loc[:, 'Positive_Review'] = df.Positive_Review.apply(lambda x: x.replace('No Positive', ''))
    df.loc[:, 'Negative_Review'] = df.Negative_Review.apply(lambda x: x.replace('No Negative', ''))
    df['text'] = df.Positive_Review + df.Negative_Review
    df["label"] = df["Reviewer_Score"].apply(lambda x: -1 if x < 7 else 1)

    return df[["text", "label"]]


def getFakeNewsDF(true_news_path, fake_news_path):
    truedf = pd.read_csv(true_news_path)
    fakedf = pd.read_csv(fake_news_path)

    truedf.drop(columns=["title", "subject", "date"], inplace=True)
    fakedf.drop(columns=["title", "subject", "date"], inplace=True)
    truedf['label'] = 1
    fakedf['label'] = -1
    df = pd.concat([truedf, fakedf]).reset_index(drop=True)

    return df


def read_vader():
    print('Indexing VADER word vectors.')

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


def dataPreparation(dataset, glove):
    tokens = []
    embeds = []
    polarities = []
    bucket = []

    for tok in dataset.keys():
        try:
            polarity = dataset[tok]
            embed = glove[tok]
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
