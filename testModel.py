import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import preprocessing as pre



LAMBDA = 1.5


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
    pipe = Pipeline([('vectorizer', TfidfVectorizer(
        max_features=num_features)), ('clf', clasiffier),])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)
    return accuracy_score(y_pred, y_test), accuracy_score(y_pred_train,
                                                          y_train), \
           pipe['vectorizer'].get_feature_names()


def test_model():
    """
    test the model on test set
    """
    df_train = pd.read_csv("train_set.csv")
    df_test = pd.read_csv("test_set.csv")
    x_train, y_train = df_train['sample'], df_train['label']
    x_test, y_test = df_test['sample'], df_test['label']
    x_train = pre.preprocessing(x_train.to_numpy())
    x_test = pre.preprocessing(x_test.to_numpy())
    model = LinearSVC(C=LAMBDA)
    max_features = int(len(x_train)/2)
    test_acc, train_acc, p = calculate_accuracies(x_train, x_test, y_train,
                                               y_test, model, max_features)
    print("number of features: ", len(p))
    print("test Accuracy: ", test_acc, " train Accuracy: ", train_acc)


test_model()