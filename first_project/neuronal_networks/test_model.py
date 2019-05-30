#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
Tool Function used to design the test model
"""

# ------------------------
#   IMPORTS
# ------------------------


# ------------------------
#   Test Function
# ------------------------
def test(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, batch_size=32)
    print(score)