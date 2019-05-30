# !/usr/bin/env python

#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
Tool Function used to design the Feature Histograms
"""

# ------------------------
#   IMPORTS
# ------------------------

import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler

# ------------------------
#   Histograms Function
# ------------------------


def histograms(dataset):
    X = dataset[:, 0:11]
    Y = dataset[:, 11]
    features = []

    X = MaxAbsScaler().fit_transform(X)

    class0 = []
    class1 = []
    class2 = []
    idx = 0

    kwargs = dict(histtype='bar', alpha=1, bins='auto', color='navy')
    plt.hist(Y, **kwargs)
    plt.title('Quality')
    plt.ylabel('Counts')
    plt.xlabel('Value')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    for feature in range(0, 11):
        idx = 0
        class0 = []
        class1 = []
        class2 = []
        for sample in Y:
            if 0 <= sample <= 4:
                class0.append(X[idx, feature])
            elif 5 <= sample <= 6:
                class1.append(X[idx, feature])
            else:
                class2.append(X[idx, feature])
            idx += 1

        kwargs0 = dict(histtype='bar', alpha=0.5, bins='auto', color='red')
        kwargs1 = dict(histtype='bar', alpha=0.5, bins='auto', color='yellow')
        kwargs2 = dict(histtype='bar', alpha=0.5, bins='auto', color='navy')

        # design plot
        plt.hist(class0, **kwargs0)
        plt.hist(class1, **kwargs1)
        plt.hist(class2, **kwargs2)
        plt.title('Feature ' + str(feature + 1) + ": " + features[feature])
        plt.ylabel('Counts')
        plt.xlabel('Value')
        plt.grid(axis='y', alpha=0.75)
        plt.legend(("Classification: 0", "Classification: 1", "Classification: 2"))
        plt.show()


if __name__ == '__main__':
    dataset = numpy.loadtxt("../csv/dataset.csv", delimiter=";", skiprows=1)
    histograms(dataset)

