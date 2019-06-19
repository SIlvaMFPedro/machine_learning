#!/usr/bin/env python3

#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
Once the model is learned, use this module to play it.
"""

# ------------------------
#   IMPORTS
# ------------------------
import numpy as np
import car_model
from neural_network import neural_network

NUM_SENSORS = 3


# ------------------------
#   Play Model Function
# ------------------------
def play(model):
    car_distance = 0
    game_state = car_model.GameState()

    # Get initial state
    _, state = game_state.frame_step(2)

    # Take action move
    while True:
        car_distance += 1
        # pick an action
        action = (np.argmax(model.predict(state, batch_size=1)))
        # take the action
        _, state = game_state.frame_step(action)

        # print values according to the action
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)


# ------------------------
#   Main Function
# ------------------------
if __name__ == "__main__":
    saved_model = 'saved-models/128-128-64-50000-25000.h5'
    model = neural_network(NUM_SENSORS, [128, 128], saved_model)
    play(model)