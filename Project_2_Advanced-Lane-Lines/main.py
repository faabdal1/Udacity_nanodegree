import numpy as np
import cv2, glob, time, pickle, os, sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from calibration import camera_calibration

# To run camera calibration and generate calibration_parameters.p
#camera_calibration(visualization = False)

# If the the calibration_parameters.p exist, reads the parameters and continues
if os.path.isfile('calibration_parameters.p'):
    mtx = pickle.load(open("calibration_parameters.p", "rb"))["mtx"]
    dist = pickle.load(open("calibration_parameters.p", "rb"))["dist"]
else:
    print ("calibration_parameters.p files does not exist, run calibration to generage the file")
    sys.exit()

# Print the distort coefficient parameters
print("Dist: " + str(dist))
print("Mtx: " + str(mtx))

print("Mtx: " + str(mtx))

