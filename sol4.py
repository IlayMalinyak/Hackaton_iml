import adaboost
import ex4_tools
import numpy as np
from plotnine import *
from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm


def boosting(X_train, y_train, X_test, y_test, T, noise):
    """
    create adaboost clasifier, train it and test
    :param X_train: train samples
    :param y_train: train lables
    :param X_test: test samples
    :param y_test: test lables
    :param T: number of weak clasifiers
    :param noise: noise ratio
    """
    boost = adaboost.AdaBoost(ex4_tools.DecisionStump, T)
    D = boost.train(X_train, y_train)
    D = D / np.max(D)*10
    err_train = np.zeros(T)
    err_test = np.zeros(T)
    min_err = 1
    min_t = 0
    desicions = np.array([5, 10, 50, 100, 200, 500])
    # q. 10, 11, 12
    for t in tqdm(range(T)):
        err_train[t] = boost.error(X_train, y_train, t)
        err_test[t] = boost.error(X_test, y_test, t)
        if err_test[t] < min_err:  # find min error
            min_err = err_test[t]
            min_t = t
        if t in desicions:  # q11
            idx = np.where(desicions == t)[0][0]
            plt.subplot(2, 3, idx + 1)
            ex4_tools.decision_boundaries(boost, X_test, y_test,
                                          num_classifiers=t)
            plt.title("T=%s, noise=%s" % (str(desicions[idx]), str(noise)))
    plt.subplot(2, 3, 6)
    ex4_tools.decision_boundaries(boost, X_test, y_test,
                                  num_classifiers=T - 1)
    plt.title("T=%s, noise=%s" % (500, str(noise)))
    plt.savefig('q11_noise_%s.jpg' % (str(noise)))

    # q12 plot
    ex4_tools.decision_boundaries(boost, X_train, y_train,
                                  num_classifiers=min_t)
    plt.title(
        "decision boundry with min error. T=%s, error=%s, noise=%s" % (min_t,
                                                                min_err,
                                                                     noise))
    plt.savefig('q12_noise_%s.jpg' % (str(noise)))

    # q10 plot
    df = pd.concat(
        [DataFrame({'T': range(1, T + 1), 'y': err_train, 'name': 'train ' \
                                                                  'error'}),
         DataFrame(
             {'T': range(1, T + 1), 'y': err_test, 'name': 'test error'})])
    p = ggplot() + geom_point(aes(x='T', y='y', color='name'), data=df,
                              size=1) \
        + labs(x='T', y='err') + \
        ggtitle('train and test errors noise=' + str(noise)) + theme(
        legend_position=(0.93, 0.6))
    ggsave(filename="q10_noise_%s.jpg" % (str(noise)), plot=p)

    # q13 plot
    ex4_tools.decision_boundaries(boost,  X_train, y_train,
                                  num_classifiers=T - 1, weights=D)
    plt.title(
        "weighted decision boundry . T=%s, test error=%s, noise=%s"
        % (T, err_test[T - 1], noise))
    plt.savefig('q13_noise_%s.jpg' % (str(noise)))


T = 500
for n in [0, 0.01, 0.4]:
    X_train, y_train = ex4_tools.generate_data(5000, n)
    X_test, y_test = ex4_tools.generate_data(200, n)
    boosting(X_train, y_train, X_test, y_test, T, n)