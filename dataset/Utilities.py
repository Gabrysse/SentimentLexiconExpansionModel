import math
import numpy as np


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
