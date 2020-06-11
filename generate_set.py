import os
import random
import pandas as pd
import numpy as np


READ = "r"
NO_SEPARATION = ""


def generate_samples(path):

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

    i = 0
    while i < len(content):
        num_of_lines = random.randint(1, 5)
        lines = content[i:i + num_of_lines]
        X.append(NO_SEPARATION.join(lines))
        y.append(file_sgn)
        i += num_of_lines


def analyze_data(path):

    df = pd.read_csv(path)
    labels = np.asarray(df.label)
    unique, counts = np.unique(labels, return_counts=True)
    ratios = counts / len(labels)

    return df, ratios


def generate_sets(path, train_ratio, path_to_train, path_to_test):

    full_set, ratios = analyze_data(path)
    train_size = int(train_ratio * len(full_set.label))
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


generate_sets("train_set.csv", 0.5, "model_train.csv",
              "model_preprocess.csv")
