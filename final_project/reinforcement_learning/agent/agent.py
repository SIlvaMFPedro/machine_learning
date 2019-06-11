#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

# ------------------------
#   IMPORTS
# ------------------------
from __future__ import print_function
import abc


# ------------------------
#   Agent Class
# ------------------------
class Agent(object):
    """
        Constructor for Agent Class
    """
    def __init__(self):
        self.__metaclass__ = abc.ABCMeta

    """
        Run Step Abstract Method
    """
    @abc.abstractmethod
    def run_step(self, measurements, sensor_data, directions, target):
        """
        Function to be redefined by the agent.
        :param measurements:
        :param sensor_data:
        :param directions:
        :param target:
        :return: carla Control Object with steering/gas/brake for the agent
        """