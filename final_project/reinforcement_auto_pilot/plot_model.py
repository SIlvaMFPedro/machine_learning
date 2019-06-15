#!/usr/bin/env python3

#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
This module takes the data in the results folder and plots the data
"""

# ------------------------
#   IMPORTS
# ------------------------
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt


# --------------------------
#   Moving Average Function
# --------------------------
def moving_average(y, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(y, window, 'same')


# --------------------------
#   Readable Output Function
# --------------------------
def readable_output(filename):
    readable = ''
    f_parts = filename.split('-')

    if f_parts[0] == 'learn_data':
        readable += 'distance: '
    else:
        readable += 'loss: '

    readable += f_parts[1] + ', ' + f_parts[2] + ' | '
    readable += f_parts[3] + ' | '
    readable += f_parts[4].split('.')[0]
    return readable


# --------------------------
#   Plot Function
# --------------------------
def plot_file(filename, type='loss'):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Turn the column into an array.
        y = []
        for row in reader:
            if type == 'loss':
                y.append(float(row[0]))
            else:
                y.append(float(row[1]))
        # Running tests should be empty.
        if len(y) == 0:
            return
        print(readable_output(filename))

        # Get the moving average so the graph isn't so crazy.
        if type == 'loss':
            window = 100
        else:
            window = 10
        y_av = moving_average(y, window)

        # Use our moving average to get some metrics.
        array = np.array(y_av)
        if type == 'loss':
            print("%f\t%f\n" % (array.min(), array.mean()))
        else:
            print("%f\t%f\n" % (array.max(), array.mean()))

        # Plot it.
        plt.clf()  # Clear.
        plt.title(filename)
        # The -50 removes an artificial drop at the end caused by the moving
        # average.
        if type == 'loss':
            plt.plot(y_av[:-50])
            plt.ylabel('Smoothed Loss')
            plt.ylim(0, 5000)
            plt.xlim(0, 250000)
        else:
            plt.plot(y_av[:-5])
            plt.ylabel('Smoothed Distance')
            plt.ylim(0, 4000)

        plt.savefig(filename + '.png', bbox_inches='tight')


# --------------------------
#   Main Function
# --------------------------
if __name__ == "__main__":
    # Get the result files.
    os.chdir("results/frames")
    # os.chdir("results/frames-valid")

    for f in glob.glob("learn*.csv"):
        plot_file(f, 'learn')

    for f in glob.glob("loss*.csv"):
        plot_file(f, 'loss')