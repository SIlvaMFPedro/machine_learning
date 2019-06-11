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
from cached_property import cached_property
import numpy as np


# ---------------------------
#   Policy Class
# ---------------------------
class PolicyOutput(object):
    """
        This stuct holds the policy output and its subproducts.
    """
    pass


# ----------------------------------
#   Sample Discrete Actions Function
# ----------------------------------
def _sample_discrete_actions(batch_probs):
    """
    Sample a batch of actions from a batch of action probabilities.
    :param batch_probs: batch_probs (ndarray): batch of action probabilities BxA
    :return: list of sampled actions
    """
    action_indices = []

    #Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    batch_probs = batch_probs - np.finfo(np.float32).epsneg

    for i in range(batch_probs.shape[0]):
        histogram = np.random.multinomial(1, batch_probs[i])
        action_indices.append(int(np.nonzero(histogram)[0]))
    return action_indices


# ----------------------------------
#   SoftmaxPolicyOutput Class
# ----------------------------------
class SoftmaxPolicyOutput(PolicyOutput):
    """
    Constructor for SoftmaxPolicyOutput class
    """
    def __init__(self, logits):
        self.logits = logits

    @cached_property
    def most_probable_actions(self):
        return np.argmax(self.probs.data, axis=1)

    @cached_property
    def probs(self):
        return functions.softmax(self.logits)

    @cached_property
    def log_probs(self):
        return functions.log_softmax(self.logits)

    @cached_property
    def action_indices(self):
        return _sample_discrete_actions(self.probs.data)

    @cached_property
    def sampled_actions_log_probs(self):
        return functions.select_item(
            self.log_probs,
            chainer.Variable(np.asarray(self.action_indices, dtype=np.int32)))

    @cached_property
    def entropy(self):
        return - functions.sum(self.probs * self.log_probs, axis=1)
