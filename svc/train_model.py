#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
Tool Function used to train model
"""

# ------------------------
#   IMPORTS
# ------------------------

from sklearn.svm import SVC

# ------------------------
#   SVC Training Function
# ------------------------


def svc_train(x_train, y_train, c, gamma, kernel):

    clf = SVC(C=c, gamma=gamma, kernel=kernel)
    clf.fit(x_train, y_train)

    return clf

