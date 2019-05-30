#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
Tool Function used to implement logistic regression
"""

# ------------------------
#   IMPORTS
# ------------------------

import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from evaluate_model import evaluate
from test_model import test
from train_model import logistic_train
from tunning_hyperparameters import logistic_tunning

# -------------------------------------
# Main Function for Logistic Regression
# -------------------------------------


if __name__ == '__main__':

    # Load Dataset
    dataset = numpy.loadtxt("../csv/dataset.csv", delimiter=";", skiprows=1)

    # Parse Data from Dataset into 3 Categories
    y = dataset[:, 11]
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

    # Now we're going to Tunning Hyperparameters only with Train Data
    best_parameters = logistic_tunning(x_train, y_train)

    # Pass the best parameters to train, and the Train Data
    trained_model_regularized = logistic_train(x_train, y_train,
                                               penalty=best_parameters["model_regularized"]["Penalty"],
                                               C=best_parameters["model_regularized"]["C"],
                                               solver=best_parameters["model_regularized"]["Solver"],
                                               multi_class=best_parameters["model_regularized"]["MultiClass"],
                                               max_iter=1000)
    trained_model_non_regularized = logistic_train(x_train, y_train,
                                                   penalty=best_parameters["model_non_regularized"]["Penalty"],
                                                   C=best_parameters["model_non_regularized"]["C"],
                                                   solver=best_parameters["model_non_regularized"]["Solver"],
                                                   multi_class=best_parameters["model_non_regularized"]["MultiClass"],
                                                   max_iter=1000)

    lg = LogisticRegression(penalty=best_parameters["model_regularized"]["Penalty"],
                            C=best_parameters["model_regularized"]["C"],
                            solver=best_parameters["model_regularized"]["Solver"],
                            multi_class=best_parameters["model_regularized"]["MultiClass"], max_iter=1000)
    cvs = cross_val_score(lg, x_train, y_train, cv=4)

    # Evaluate the model
    print("\n Evaluate the model \n")
    print("\nRegularized:")
    evaluate(trained_model_regularized, x_train, y_train)
    print("\n Cross_validation")
    print(cvs)
    print("\nNon Regularized:")
    evaluate(trained_model_non_regularized, x_train, y_train)

    # Test the model
    print("\n Test the model \n")
    print("\nRegularized:")
    test(trained_model_regularized, x_test, y_test)
    print("\nNon Regularized:")
    test(trained_model_non_regularized, x_test, y_test)