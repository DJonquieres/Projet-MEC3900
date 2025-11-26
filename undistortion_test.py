import cv2
import numpy as np
from zhang_calibration import *

def straightness_error(pts_row):
    xs = pts_row[:, 0]
    ys = pts_row[:, 1]
    a, b = np.polyfit(xs,ys, 1)
    y_fit = a*xs +b
    return np.sqrt(np.mean((ys - y_fit)**2))

def test_error(path, path_test, CHECKERBOARD, ):


    ret, K, dist, rvecs, tvecs = calibration(path, CHECKERBOARD)

    img = cv2.imread(path_test)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        corners = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30, 0.001))
    
        undistorted_pts = cv2.undistortPoints(corners, K, dist, P=K)
        undistorted_pts = undistorted_pts.reshape(-1,2)

        row0_orig = corners[0:CHECKERBOARD[0], 0, :]
        row0_und = undistorted_pts[0:CHECKERBOARD[0], :]

        error_orig = straightness_error(row0_orig)
        error_und = straightness_error(row0_und)

    return ret, error_orig, error_und

CHECKERBOARD = (9,6)
path1 = "calib_images/cam1"
path2 = "calib_images/cam2"
path_test1 = "path/to/testimg.jpg"
path_test2 = "path/to/testimg2.jpg"
ret1, error_orig1, error_und1 = test_error(path1, path_test1, CHECKERBOARD)
ret2, error_orig2, error_und2 = test_error(path2, path_test2, CHECKERBOARD)


if ret1 and ret2:
    print(f"Straightness error 1 (original): {error_orig1}px")
    print(f"Straightness error 1 (undistorted): {error_und1}px")
    print(f"Straightness error 2 (original): {error_orig2}px")
    print(f"Straightness error 2 (undistorted): {error_und2}px")    