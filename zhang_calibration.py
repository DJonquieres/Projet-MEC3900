import cv2
import numpy as np 
import glob
def calibration(path, CHECKERBOARD=(9,6)):

    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)

    objpoints = []  #3D Points
    imgpoints = []  #2D Points

    images = glob.glob(path+"/*.jpg")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11,11), (-1,-1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return (ret,K, dist, rvecs, tvecs)

if __name__ == '__main__':

    CHECKERBOARD = (9, 6)

    path1 = "calib_images/cam1"
    path2 = "calib_images/cam2"
    ret1, K1, dist1, rvecs1, tvecs1 = calibration(path1, CHECKERBOARD)
    ret2, K2, dist2, rvecs2, tvecs2 = calibration(path2, CHECKERBOARD)

    print('Intrinsic matrix K1:')
    print(K1)
    print('Intrinsic matrix K2:')
    print(K2)
    print('n/Distortion coefficients dist1 (k1, k2, p1, p2, k3):')
    print(dist1.ravel())
    print('n/Distortion coefficients dist2 (k1, k2, p1, p2, k3):')
    print(dist2.ravel())

# img = cv2.imread("path/to/testimg.jpg")
# img2 = cv2.imread("path/to/testimg2.jpg")
# h,w = img.shape[:2]
# new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w,h))

# undistorted = cv2.undistort(img, K, dist, None, new_K)
# cv2.imwrite("undistorted2.jpg", undistorted)
