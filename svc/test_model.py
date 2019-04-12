#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
Tool Function used to test model
"""

# ------------------------
#   IMPORTS
# ------------------------

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# ------------------------
#   Test Function
# ------------------------


def test(model, x_test, y_test):
    predicted_data = model.predict(x_test)

    print("Mean squared error - test set:", mean_squared_error(y_test, predicted_data))
    print("Coefficient of determination - test set:", r2_score(y_test, predicted_data))
    print("Accuracy - test set:", accuracy_score(y_test, predicted_data))