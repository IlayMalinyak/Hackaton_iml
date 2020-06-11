import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def calculate_accuracies(X_train, X_test, y_train, y_test,
                         classifier, num_features=None):
    """
    calculate test and train accuracy
    :param X_train: train sample
    :param X_test: test sample
    :param y_train: train lables
    :param y_test: test lables
    :param classifier: classifier learner
    :param num_features: max features for word embedding
    :return: test and train accuracy
    """
    X_train, y_train = X_train, y_train
    X_test, y_test =  X_test, y_test
    pipe = Pipeline([('vectorizer', TfidfVectorizer(max_features=num_features)),
                   ('clf', classifier),
                  ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)
    return accuracy_score(y_pred, y_test), accuracy_score(y_pred_train,
                                                          y_train)


def test_svm(file, Cs):
    """
    apply cross validation with linear soft SVM
    :param file: path to train\validation set
    :param Cs: array of lambdas
    """
    df_train = pd.read_csv(file)
    X_train, X_test, y_train, y_test = train_test_split(df_train['sample'], df_train['label'],
                                                           test_size=0.2)
    acc_test = []
    acc_train = []
    for c in tqdm(Cs):
        model = LinearSVC(C=c)
        acc_test.append(calculate_accuracies(X_train, X_test, y_train, y_test,
                                       model, 20000)[0])
        acc_train.append(calculate_accuracies(X_train, X_test, y_train, y_test,
                                             model, 20000)[1])
    c_max = np.argmax(acc_test)
    plt.plot(Cs, acc_test, label='Test')
    plt.plot(Cs, acc_train, label='Train')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("Accuracy")
    plt.title('SVM Accuracy vs %s. %s max = %f' % (r'$\lambda$',
                                                   r'$\lambda$', Cs[c_max]))
    plt.legend()
    plt.savefig("svm_cv_validation")
    plt.show()


def test_regression(file, Cs):
    """
    apply cross validation with logistic regression
    :param file: path to train\validation set
    :param Cs: array of lambdas
    """
    df_train = pd.read_csv(file)
    X_train, X_test, y_train, y_test = train_test_split(df_train['sample'], df_train['label'],
                                                           test_size=0.2)
    acc_test = []
    acc_train = []
    for c in tqdm(Cs):
        model = LogisticRegression(C=c, solver='saga')
        acc_test.append(calculate_accuracies(X_train, X_test, y_train, y_test,
                                       model, 20000)[0])
        acc_train.append(calculate_accuracies(X_train, X_test, y_train, y_test,
                                             model, 20000)[1])
    c_max = np.argmax(acc_test)
    plt.plot(Cs, acc_test, label='Test')
    plt.plot(Cs, acc_train, label='Train')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("Accuracy")
    plt.title('logistic regression Accuracy vs %s. %s max = %f' % (
        r'$1/\lambda$',  r'$\lambda$', 1/(Cs[c_max])))
    plt.legend()
    plt.savefig("logistic_cv")
    plt.show()


# test both models and chose the better one
test_svm("validation_train.csv", np.linspace(0.1, 3, 30))
test_regression("model_preprocess.csv", np.linspace(0.1, 5, 30))