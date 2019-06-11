#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

# USAGE
# python track_object.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video input/race.mp4 \
#	--label person --output output/race_output.avi

# ------------------------
#   IMPORTS
# ------------------------
import numpy as np
import argparse
import cv2
