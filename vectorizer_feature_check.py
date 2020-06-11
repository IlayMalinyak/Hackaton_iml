import pandas as pd
from pandas import DataFrame
import numpy as np
from plotnine import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import preprocessing as pre


def calculate_accuracies(X_train, X_test, y_train, y_test, num_features=None):
    """
    This function calculates the accuracy of a basic Multinomial Naive Bayes classifier based on the
     number of features for the vectorizer.
    :param X_train: The training samples
    :param X_test: The test samples
    :param y_train: The training labels
    :param y_test: The test labels
    :param num_features: The number of features to test in the vectorizer
    :return: The accuracy score of the model
    """
    pipe = Pipeline([('vectorizer', TfidfVectorizer(max_features=num_features)),
                  ('classifier', MultinomialNB())])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return accuracy_score(y_pred, y_test)


def plot_accuracies(file_train, features):
    """
    Plots the accuracies of a basic Multinomial Naive Bayes classifier as a function of the number
    of features in the Vectorizer
    :param file_train: The file to read the train data from
    :param features: a list consisting of different number of features
    """
    df_train = pd.read_csv(file_train) # reads the file into df
    # creates random train-test-split
    X_train, X_test, y_train, y_test = train_test_split(df_train['sample'], df_train['label'],
                                                        test_size=0.2)
    # apply preproccessing
    X_train = pre.preprocessing(X_train.to_numpy())
    X_test = pre.preprocessing(X_test.to_numpy())
    acc = np.zeros(len(features))
    # loop on the number of features
    for i in range(len(features)):
        acc[i] = calculate_accuracies(X_train, X_test, y_train, y_test, features[i])
    # plot the results
    df_1 = DataFrame({"features": features, "acc":acc})
    plot = (ggplot(df_1) + geom_line(aes(x="features", y="acc")) +
                 labs(title="accuracies on train test based on num_feat", x="features",
                      y="accuracy"))
    print(plot)
