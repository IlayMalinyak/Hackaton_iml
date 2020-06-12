"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Auther(s): Ilay malinyak, Yasmin Selah, Gal Fabelzon, Or Gershoni

===================================================
"""
# import preprocessing as pre
import joblib
import nltk
from sklearn.pipeline import Pipeline


class GitHubClassifier:
    def __init__(self):
        # load trained model
        self.model = joblib.load("Git_classifier.pkl")
        self.most_common = 'else'

    def classify(self, X):
        """
        Receives a list of m unclassified pieces of code, and predicts for each
        one the Github project it belongs to.
        :param X: a numpy array of shape (m,) containing the code segments (strings)
        :return: y_hat - a numpy array of shape (m,) where each entry is a number between 0 and 6
        0 - building_tool
        1 - espnet
        2 - horovod
        3 - jina
        4 - PuddleHub
        5 - PySolFC
        6 - pytorch_geometric
        """
        X = self.preprocessing(X)
        y_hat = self.model.predict(X)
        return y_hat


    def clean_sentence(self, string):
        """
        clean sentence from panctuations
        :param string:  sentence
        :return: cleaned sentence
        """
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        return " ".join(tokenizer.tokenize(string)).replace('\n', '')


    def clean_sentence_most_common(self, string):
        """
        clean sentence from panctuations and most common word
        :param string:  sentence
        :return: cleaned sentence
        """
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        string = " ".join(tokenizer.tokenize(string)).replace('\n', '')
        string = string.replace(self.most_common, '')
        return string


    def preprocessing(self, np_strings_array, clean_most_common=False):
        """
        This function gets the np array of strings and doing the preprocessing.
        :param np_strings_array: the arrray of string from the input project file
        :param clean_most_common: a flag
        :return: the cleaned array
        """
        if clean_most_common == True:
            clean_data = list(
                map(self.clean_sentence_most_common, np_strings_array))
        else:
            clean_data = list(map(self.clean_sentence, np_strings_array))
        return clean_data
