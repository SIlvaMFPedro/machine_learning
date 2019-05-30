#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
Tool Function used to implement neuronal network
"""

# ------------------------
#   IMPORTS
# ------------------------

import numpy
from sklearn.model_selection import train_test_split
from keras.models import load_model
from evaluate_data_model import evaluate
from optimize import optimize
from test_model import test
from train_model import nn_train

# ---------------------------------------------------
#   Main Function for Neuronal Network Implementation
# ---------------------------------------------------
if __name__ == '__main__':

    # Load Dataset
    dataset = numpy.loadtxt("../csv/dataset.csv", delimiter=";", skiprows=1)

    # Parse Data from Dataset into 3 Categories
    y = dataset[:, 11]
    classifications = 3
    new_y = []
    for sample in y:
        if 0 <= sample <= 4:
            # 0, 1, 2, 3, 4
            new_y.append(0)
        elif 5 <= sample <= 6:
            # 5, 6
            new_y.append(1)
        else:
            # 7, 8, 9, 10
            new_y.append(2)
    y = new_y
    X = dataset[:, 0:11]

    # Divide Dataset: 20% Test and 80% Train
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    x_train_60, x_val, y_train_60, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=5)

    # best_parameters
    best_parameters = optimize(x_train_60, y_train_60, x_val, y_val)

    # Pass the best parameters to train, and the Train Data
    trained_model = nn_train(x_train, y_train, x_test, y_test, best_parameters)

    trained_model.save('final_model.h5')

    # trained_model = load_model('final_model.h5')

    # Evaluate the model
    evaluate(trained_model, x_train, y_train)

    # Test the model
    test(trained_model, x_test, y_test)