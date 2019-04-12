#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
Tool Function used to design the train model
"""

# ------------------------
#   IMPORTS
# ------------------------

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# ----------------------------------
#   Neural Network Training Function
# ----------------------------------


def nn_train(x_train, y_train, x_test, y_test, best_parameters):

    classifications = 3

    model = Sequential()

    model.add(Dense(12, input_dim=11, activation='relu'))

    for _ in range(best_parameters["dense_layer"]):
        model.add(Dense(best_parameters["layer_size"], activation="relu"))

    model.add(Dense(8, activation='relu'))
    model.add(Dense(classifications, activation="softmax"))

    adam = optimizers.adam(lr=0.01, clipnorm=1.)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy', 'mse'])

    model.fit(x_train, y_train, epochs=500, batch_size=50, verbose=2, validation_data=(x_test, y_test))

    return model
