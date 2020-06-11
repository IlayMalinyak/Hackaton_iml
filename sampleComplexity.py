import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix


def calculate_accuracies(X_train, X_test, y_train, y_test,
                         clasiffier, num_features=None):
    """
    calculate test and train accuracy
    :param X_train: train sample
    :param X_test: test sample
    :param y_train: train lables
    :param y_test: test lables
    :param clasiffier: clasiffier learner
    :param num_features: max features for word embedding
    :return: test, train accuracy and list of features names
    """

    pipe = Pipeline([('vectorizer', TfidfVectorizer(max_features=num_features)),
                   ('classifier', clasiffier),
                  ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)
    return accuracy_score(y_pred, y_test), accuracy_score(y_pred_train,
                                                          y_train)


def calc_acc(file):
    """
    calculate accuracy as fanction of model complexity
    :param file: path to train set
    """
    df_train = pd.read_csv(file)
    X_train, X_test, y_train, y_test = train_test_split(df_train['sample'], df_train['label'],
                                                               test_size=0.2)
    length = len(X_train)
    acc_test = []
    acc_train = []
    c = 1.5
    sample_size = np.linspace(1000, length, 10)
    for l in tqdm(sample_size):
        model = LinearSVC(C=c)
        x_train = X_train.iloc[:int(l)]
        Y_train = y_train.iloc[:int(l)]
        features = min(20000, int(l/2))
        acc_test.append(calculate_accuracies(x_train, X_test,
                                             Y_train,
                                             y_test,
                                             model, features)[0])
        acc_train.append(calculate_accuracies(x_train, X_test,
                                              Y_train,
                                              y_test,
                                              model, features)[1])

    plt.plot(sample_size, acc_test, label='Test')
    plt.plot(sample_size, acc_train, label='Train')
    plt.xlabel('sample size')
    plt.ylabel("Accuracy")
    plt.title('SVM Accuracy vs sample size')
    plt.legend()
    plt.savefig("svm_sample_complexity_val")
    plt.show()


calc_acc("validation_train.csv")