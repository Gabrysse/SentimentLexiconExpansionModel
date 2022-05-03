import numpy as np
from nltk import RegexpTokenizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer

tokNot = RegexpTokenizer(r'not \w+|\w+')


def tok(doc):
    return tokNot.tokenize(doc.lower().replace("n't", " not"))


def seed_regression(dataframe):
    vectorizer = CountVectorizer(tokenizer=tok, min_df=25)
    #   vectorizer = TfidfVectorizer(tokenizer=tok, use_idf=False, min_df=50)

    regression = Ridge()

    pipe = Pipeline([
        ('cv', vectorizer),
        ('lr', regression)
    ])

    pipe.fit(dataframe.reviewText, dataframe.overall)
    return vectorizer, regression


def seed_filter(dataframe, vectorizer, regression, frequency=500):
    a = vectorizer.fit_transform(dataframe.reviewText)
    mask = np.array(a.sum(axis=0) > frequency).squeeze()
    tokens = vectorizer.get_feature_names_out()[mask]
    coefs = regression.coef_[mask]
    return dict(zip(tokens, coefs))
