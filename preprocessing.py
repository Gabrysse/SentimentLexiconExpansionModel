import numpy as np
from nltk import RegexpTokenizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

tokNot = RegexpTokenizer(r'not \w+|\w+')


def tok(doc):
    return tokNot.tokenize(doc.lower().replace("n't", " not"))


def get_token_counts(reviews):
    print("Calculating token count...")
    vectorizer = CountVectorizer(tokenizer=tok, min_df=25)
    X = vectorizer.fit_transform(reviews)
    return X, vectorizer.get_feature_names_out()


def train_linear_model(X, y):
    print("Training linear model...")
    regression = Ridge()
    regression.fit(X, y)
    return regression.coef_


def seed_filter2(X, features, coeff, frequency=500):
    print("Seed filtering...")
    mask = np.array(X.sum(axis=0) > frequency).squeeze()
    tokens = features[mask]
    coefs = coeff[mask]
    return dict(zip(tokens, coefs))


def seed_regression(dataframe):
    print("Seed regression...")
    vectorizer = CountVectorizer(tokenizer=tok, min_df=25)
    #   vectorizer = TfidfVectorizer(tokenizer=tok, use_idf=False, min_df=50)

    regression = Ridge()
    # w_negative = len(dataframe['overall'][dataframe['overall'] == +1]) / len(dataframe['overall'])
    # w_positive = 1 - w_negative
    # svm = LinearSVC(random_state=0, fit_intercept=False, class_weight={-1: w_negative, 1: w_positive}, max_iter=5000)

    pipe = Pipeline([
        ('cv', vectorizer),
        ('lr', regression)
    ])

    pipe.fit(dataframe.reviewText, dataframe.overall)
    return vectorizer, regression


def seed_filter(dataframe, vectorizer, regression, frequency=500):
    print("Seed filtering...")
    a = vectorizer.fit_transform(dataframe.reviewText)
    mask = np.array(a.sum(axis=0) > frequency).squeeze()
    tokens = vectorizer.get_feature_names_out()[mask]
    coefs = regression.coef_[mask]
    return dict(zip(tokens, coefs))
