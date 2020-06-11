import nltk
import pandas as pd
from pandas import DataFrame
from plotnine import*
import numpy as np
from matplotlib import pyplot as plt


def clean_sentence(string):
    """
    clean snetence from panctuations
    :param string:  sentence
    :return: cleaned sentence
    """
    tokenizer = nltk.RegexpTokenizer(r"\w+\s+")
    return " ".join(tokenizer.tokenize(string)).replace('\n', '')


def data_summary(name):
    """
    plot word frequency plot and return dataframe of 30's most common words
    :param name: name of repository
    :return: dataframe with words frequencies
    """
    path = "data/%s_all_data.txt" % name
    with open(path, 'r', encoding='utf-8') as myfile:
        data = myfile.readlines()
        clean_data = list(map(clean_sentence, data))
        fdist = plot_freq(clean_data, name)
        common = fdist.most_common(30)
        words = [word[0].split('\n')[0] for word in common]
        freq = [word[1] for word in common]
        df = DataFrame({"words":words, "x":np.arange(len(words)),
                        "frequency":freq, "name":name})
        return df


def plot_freq(clean_data, name):
    """
    plot 10's most common words
    :param clean_data: data of words
    :param name: name of repository
    :return: freqDist object
    """
    fig = plt.figure(figsize=(10, 4))
    plt.gcf().subplots_adjust(bottom=0.15)  # to avoid x-ticks cut-off
    fdist = nltk.FreqDist(clean_data)
    fdist.plot(11, cumulative=False, title="10's frequency %s dataset" % name)
    plt.show()
    fig.savefig("words_frequency_10_%s" %name, bbox_inches="tight")
    return fdist



names = ["building_tool", "espnet","horovod","jina","PaddleHub", "PySolFC",
         "pytorch_geometric"]
dfs = []
for name in names:
    dfs.append(data_summary(name))
df = pd.concat(dfs)
df = data_summary("data/PaddleHub_all_data.txt")
p = ggplot(df) + geom_line(aes(x='x', y='frequency', color='name')) + \
    ggtitle("30 most common words frequency")
ggsave(p, "common_words_distribution")
print(p)
