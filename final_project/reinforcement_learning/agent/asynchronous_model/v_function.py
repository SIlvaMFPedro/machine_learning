#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

# ------------------------
#   IMPORTS
# ------------------------
import chainer
from chainer import functions
from chainer import links


# ---------------------------
#   VFunction Class
# ---------------------------
class VFunction(object):
    pass


# ---------------------------
#   FCVFunction Class
# ---------------------------
class FCVFunction(chainer.ChainList, VFunction):
    """
        Constructor for FCVFunction class
    """
    def __init__(self, n_input_channels, n_hidden_layers=0,
                 n_hidden_channels=None):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        layers = []
        if n_hidden_layers > 0:
            layers.append(links.Linear(n_input_channels, n_hidden_channels))
            for i in range(n_hidden_layers - 1):
                layers.append(links.Linear(n_hidden_channels, n_hidden_channels))
            layers.append(links.Linear(n_hidden_channels, 1))
        else:
            layers.append(links.Linear(n_input_channels, 1))

        super(FCVFunction, self).__init__(*layers)

    """
        Callback function
    """
    def __call__(self, state):
        h = state
        for layer in self[:-1]:
            h = functions.relu(layer(h))
        h = self[-1](h)
        return h
