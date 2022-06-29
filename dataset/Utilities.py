import gzip
import json
import math
import numpy as np
import pandas as pd


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getAmazonDF(path):
    i = 0
    df = {}
    print("Reading review dataset...")
    for d in parse(path):
        df[i] = d
        i += 1

    df = pd.DataFrame.from_dict(df, orient='index')
    df = df[df['unixReviewTime'] < 1406851200]  # remove the reviews
    df = df.drop(
        columns=["verified", "reviewTime", "reviewerID", "asin", "reviewerName", "summary", "unixReviewTime", "vote",
                 "style", "image"])
    df = df[df["overall"] != 3]
    df['overall'] = df.apply(lambda row: (row["overall"] > 3) * 2 - 1, axis=1)
    df = df.fillna("")
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
