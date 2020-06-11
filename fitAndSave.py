import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing as pre
import joblib

LAMBDA = 1.5


def fit_and_save():
    """
    fit model on the full data and save
    """
    df_train = pd.read_csv("all_data.csv")
    x_train, y_train = df_train['sample'], df_train['label']
    x_train = pre.preprocessing(x_train.to_numpy())
    classifier = LinearSVC(C=LAMBDA)
    max_features = int(len(x_train)/2)
    pipe = Pipeline([('vectorizer', TfidfVectorizer(
        max_features=max_features)), ('classifier', classifier), ])
    joblib.dump(pipe.fit(x_train, y_train), 'Git_classifier.pkl')


fit_and_save()