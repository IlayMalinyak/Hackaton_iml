import os
import random
import pandas as pd
import numpy as np

READ = "r"
NO_SEPARATION = ""


def generate_samples(path):
    """
    :param path: path of directory containing .txt project files
    :return: X vector containing samples (each sample is a string of 1-5
    lines of codes), y - response vector (values 0-6 according to project)
    """

    X, y = [], []
    file_sgn = 0

    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        f = open(full_path, 'r', encoding="utf8")
        content = f.readlines()
        split_randomly(X, y, content, file_sgn)
        f.close()
        file_sgn += 1

    return X, y


def split_randomly(X, y, content, file_sgn):
    """
    Filling X,y vectors with samples and labels
    :param X: 1d vector of samples to fill
    :param y: 1d vector of labels to fill
    :param content: array of strings to split
    :param file_sgn: sign of file that content was taken from
    :return:
    """

    i = 0
    while i < len(content):
        num_of_lines = random.randint(1, 5)
        lines = content[i:i + num_of_lines]
        X.append(NO_SEPARATION.join(lines))
        y.append(file_sgn)
        i += num_of_lines


def analyze_data(df):
    """
    return frequencies of each label (0-6) inside the df
    :param df: data frame with labels 0-6
    :return:
    """

    labels = np.asarray(df.label)
    unique, counts = np.unique(labels, return_counts=True)
    ratios = counts / len(labels)

    return ratios


def generate_sets(path, train_ratio, path_to_train, path_to_test):
    """
    :param path: pathname to .csv file
    :param train_ratio: required ratio to extract as train set
    :param path_to_train: path of train df to be saved
    :param path_to_test: path of test df to be saved
    :return:
    """

    full_set = pd.read_csv(path)
    ratios = analyze_data(full_set)
    train_size = int(train_ratio * len(full_set.label))
    # determine the amount of samples to be taken as test samples for each
    # label
    samples_amount = (ratios * train_size).astype(int)
    X_train, y_train, X_test, y_test = [], [], [], []

    for file_sgn, df_file in full_set.groupby(['label']):

        lines = np.asarray(df_file['sample'])
        indices_train = np.random.choice(len(lines), samples_amount[file_sgn], replace=False)
        test_samples = np.delete(lines, indices_train)
        train_samples = np.take(lines, indices_train)

        X_train.extend(train_samples)
        X_test.extend(test_samples)
        y_train.extend([file_sgn] * len(train_samples))
        y_test.extend([file_sgn] * len(test_samples))

    train_set = pd.DataFrame({'sample': X_train, 'label': y_train})
    test_set = pd.DataFrame({'sample': X_test, 'label': y_test})

    train_set.to_csv(path_or_buf=path_to_train, index=False)
    test_set.to_csv(path_or_buf=path_to_test, index=False)


generate_sets("train_set.csv", 0.8, "model_train.csv",
              "model_preprocess.csv")
