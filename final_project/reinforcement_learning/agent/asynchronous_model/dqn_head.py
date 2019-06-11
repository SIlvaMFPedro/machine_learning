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
from . import nonlinearity


# ---------------------------
#   NatureDQNHead Class
# ---------------------------
class NatureDQNHead(chainer.ChainList):
    """
        Constructor for NatureDQNHead class
    """

    def __init__(self, n_input_channels=None, n_output_channels=512,
                 nonlinearity_str=None, bias=None):
        self.n_input_channels = n_input_channels
        self.nonlinearity = nonlinearity.get_from_str(nonlinearity_str)
        self.n_output_channels = n_output_channels

        layers = [
            links.Convolution2D(n_input_channels, 32, 8, stride=4, bias=bias),
            links.Convolution2D(32, 64, 4, stride=2, bias=bias),
            links.Convolution2D(64, 64, 3, stride=1, bias=bias),
            links.Linear(3136, n_output_channels, bias=bias),
        ]

        super(NatureDQNHead, self).__init__(*layers)

    """
        Callback function for NatureDQNHead class
    """
    def __call__(self, state):
        h = state
        for layer in self:
            h = self.nonlinearity(layer(h))
        return h


# ---------------------------
#   NIPSDQNHead Class
# ---------------------------
class NIPSDQNHead(chainer.ChainList):
    """
        Constructor for NIPSDQNHead class
    """
    def __init__(self, n_input_channels=None, n_output_channels=256,
                 nonlinearity_str=None, bias=None):
        self.n_input_channels = n_input_channels
        self.nonlinearity = nonlinearity.get_from_str(nonlinearity_str)
        self.n_output_channels = n_output_channels

        layers = [
            links.Convolution2D(n_input_channels, 16, 8, stride=4, bias=bias),
            links.Convolution2D(16, 32, 4, stride=2, bias=bias),
            links.Linear(2592, n_output_channels, bias=bias),
        ]

        super(NIPSDQNHead, self).__init__(*layers)

    """
        Callback function for NIPSDQNHead class
    """
    def __call__(self, state):
        h = state
        for layer in self:
            h = self.nonlinearity(layer(h))
        return h
