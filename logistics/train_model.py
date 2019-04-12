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

from sklearn.linear_model import LogisticRegression

# ----------------------------------
# Logistic Regression Train Function
# ----------------------------------


def logistic_train(x_train, y_train, penalty, C, solver, multi_class, max_iter):
    lg = LogisticRegression(penalty=penalty, C=C, solver=solver, multi_class=multi_class, max_iter=max_iter)
    lg.fit(x_train, y_train)
    return lg
