import numpy as np
from nltk import RegexpTokenizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer

tokNot = RegexpTokenizer(r'not \w+|\w+')


def tok(doc):
    """
    Function used to tokenize text
        :param doc: Document you want to tokenize
        :return: list containing the token of the given document
    """
    return tokNot.tokenize(doc.lower().replace("n't", " not"))


def get_token_counts(reviews):
    """
    Function used to obtain the BoW representation of each document in input.
        :param reviews: reviews that you want to process
        :return: BoW vector, List of features names
    """
    print("🔢 Calculating token count...")
    vectorizer = CountVectorizer(tokenizer=tok, min_df=25)
    X = vectorizer.fit_transform(reviews)
    return X, vectorizer.get_feature_names_out()


def train_linear_model(X, y):
    """
    Function used to train the linear regression model.
        :param X: X
        :param y: y
        :return: Regression coefficient
    """
    print("🏃🏻 Training linear model...")
    regression = Ridge()
    regression.fit(X, y)
    return regression.coef_


def seed_filter(X, features, coeff, frequency=500):
    """
    Function used to filter seed data.
        :param X: BoW vector
        :param features: List of features names
        :param coeff: Regression coefficent
        :param frequency: Min frequency
        :return: Dictonary containing {token: coefficent}
    """
    print("🔍 Seed filtering...\n")
    mask = np.array(X.sum(axis=0) > frequency).squeeze()
    tokens = features[mask]
    coefs = coeff[mask]
    return dict(zip(tokens, coefs))
