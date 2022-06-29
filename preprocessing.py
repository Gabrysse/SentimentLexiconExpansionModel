import numpy as np
from nltk import RegexpTokenizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

tokNot = RegexpTokenizer(r'not \w+|\w+')


def tok(doc):
    return tokNot.tokenize(doc.lower().replace("n't", " not"))


def seed_regression(dataframe):
    vectorizer = CountVectorizer(tokenizer=tok, min_df=25)
    #   vectorizer = TfidfVectorizer(tokenizer=tok, use_idf=False, min_df=50)

    # regression = Ridge()
    svm = LinearSVC(random_state=0, tol=1e-4)

    pipe = Pipeline([
        ('cv', vectorizer),
        ('lr', svm)
    ])

    pipe.fit(dataframe.reviewText, dataframe.overall)
    return vectorizer, svm


def seed_filter(dataframe, vectorizer, regression, frequency=500):
    a = vectorizer.fit_transform(dataframe.reviewText)
    mask = np.array(a.sum(axis=0) > frequency).squeeze()
    tokens = vectorizer.get_feature_names_out()[mask]
    coefs = regression.coef_[mask]
    return dict(zip(tokens, coefs))
