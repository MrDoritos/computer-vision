#!/bin/python3

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
source_directory = './calibration_images'

imgL = cv.imread(source_directory + '/left/left_0.jpg', cv.IMREAD_GRAYSCALE)
imgR = cv.imread(source_directory + '/right/right_0.jpg', cv.IMREAD_GRAYSCALE)
 
stereo = cv.StereoBM.create(numDisparities=16, blockSize=51)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()