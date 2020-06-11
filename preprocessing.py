import nltk
# import pandas as pd
from pandas import DataFrame
# from plotnine import*
import numpy as np
# from matplotlib import pyplot as plt


def clean_sentence(string):
    """
    clean sentence from panctuations
    :param string:  sentence
    :return: cleaned sentence
    """
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    return " ".join(tokenizer.tokenize(string)).replace('\n', '')


def clean_sentence_most_common(string):
    """
    clean sentence from panctuations and most common word
    :param string:  sentence
    :return: cleaned sentence
    """
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    string = " ".join(tokenizer.tokenize(string)).replace('\n', '')
    string = string.replace('else', '')
    return string


def preprocessing(np_strings_array, clean_most_common=False):
    """
    This function gets the np array of strings and doing the preprocessing.
    :param np_strings_array: the arrray of string from the input project file
    :param clean_most_common: a flag
    :return: the cleaned array
    """
    if clean_most_common==True:
        clean_data = list(map(clean_sentence_most_common, np_strings_array))
    else:
        clean_data =list(map(clean_sentence, np_strings_array))
    return clean_data

if __name__ == '__main__':
    with open('mini_test.csv', 'r', encoding='utf-8') as myfile:
        data = myfile.readlines()[1:]
        np_strings_array = np.array(data)
        preprocessing(np_strings_array, True)