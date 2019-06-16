#!/usr/bin/env python3

#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
This module takes the data in the evaluation scores folder and plots the data
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

    readable += 'evaluation: '
    readable += f_parts[1] + ', ' + f_parts[2] + ' | '
    readable += f_parts[3] + ' | '
    readable += f_parts[4].split('.')[0]
    return readable


# --------------------------
#   Plot Function
# --------------------------
def plot_file(filename, type='loss'):
    with open(filename, 'r') as csvfile:
        print(readable_output(filename))
        reader = csv.reader(csvfile)
        # Turn the column into an array.
        x = [25000, 50000, 75000, 100000]
        y = []
        for row in reader:
            y.append(float(row[0]))
        # Running tests should be empty.
        if len(x) == 0 or len(y) == 0:
            return

        # Plot it.
        plt.clf()  # Clear.
        plt.title(filename)
        plt.plot(x, y, marker='o')
        plt.xlabel("Number of Frames")
        plt.ylabel("Evaluation Score")

        plt.savefig(filename + '.png', bbox_inches='tight')

# --------------------------
#   Main Function
# --------------------------
if __name__ == "__main__":
    # Get the result files.
    os.chdir("evaluation-scores")
    # os.chdir("results/frames-valid")

    for f in glob.glob("evaluation*.csv"):
        plot_file(f, 'evaluation')

