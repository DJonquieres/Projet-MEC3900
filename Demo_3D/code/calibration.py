import sys
import numpy as np
import time
import imutils
import cv2

cv_file = cv2.FileStorage()
cv_file.open('../stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

print("L_x:", None if stereoMapL_x is None else stereoMapL_x.shape, stereoMapL_x.dtype)
print("L_y:", None if stereoMapL_y is None else stereoMapL_y.shape, stereoMapL_y.dtype)
print("R_x:", None if stereoMapR_x is None else stereoMapR_x.shape, stereoMapR_x.dtype)
print("R_y:", None if stereoMapR_y is None else stereoMapR_y.shape, stereoMapR_y.dtype)

def undistortRectify(frameL, frameR):

    undistortedL = cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LINEAR)
    undistortedR = cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LINEAR)

    return undistortedL, undistortedR