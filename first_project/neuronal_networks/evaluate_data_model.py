#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
Tool Function used to evaluate data model
"""

# ------------------------
#   IMPORTS
# ------------------------

# ------------------------
#   Evaluate Function
# ------------------------


def evaluate(model, x_train, y_train):
    score = model.evaluate(x_train, y_train, batch_size=32)
    print(score)