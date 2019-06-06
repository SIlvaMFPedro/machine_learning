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
from . import policy_output
from logging import getLogger
logger = getLogger(__name__)


# ---------------------------
#   Policy Class
# ---------------------------
class Policy(object):
    """
        Abstract Policy Class
    """
    def __call__(self, state):
        raise NotImplementedError


# ---------------------------
#   SoftmaxPolicy Class
# ---------------------------
class SoftmaxPolicy(Policy):
    """
        Abstract Softmax Policy Class
    """
    def compute_logits(self, state):
        """
        :param state:
        :return: ~chainer.Variable: logits of actions
        """
        raise NotImplementedError

    """
        Callback method
    """
    def __call__(self, state):
        return policy_output.SoftmaxPolicyOutput(self.compute_logits(state))


# ---------------------------
#   SoftmaxPolicy Class
# ---------------------------
class FCSoftmaxPolicy(chainer.ChainList, SoftmaxPolicy):
    """Softmax policy that consists of FC layers and rectifiers"""

    """
        FCSoftmaxPolicy constructor
    """
    def __init__(self, n_input_channels, n_actions,
                 n_hidden_layers=0, n_hidden_channels=None):
        self.n_input_channels = n_input_channels
        self.n_actions = n_actions
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        layers = []
        if n_hidden_layers > 0:
            layers.append(L.Linear(n_input_channels, n_hidden_channels))
            for i in range(n_hidden_layers - 1):
                layers.append(L.Linear(n_hidden_channels, n_hidden_channels))
            layers.append(L.Linear(n_hidden_channels, n_actions))
        else:
            layers.append(L.Linear(n_input_channels, n_actions))

        super(FCSoftmaxPolicy, self).__init__(*layers)

    """
        Compute logits method
    """
    def compute_logits(self, state):
        h = state
        for layer in self[:-1]:
            h = F.relu(layer(h))
        h = self[-1](h)
        return h


# ---------------------------
#   GaussianPolicy Class
# ---------------------------
class GaussianPolicy(Policy):
    """Abstract Gaussian policy class.
    """
    pass
