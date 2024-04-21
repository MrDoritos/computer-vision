#!/bin/python3
import numpy as np
import cv2
import glob

# Chessboard size
chessboard_size = (9, 6)  # inner corners

# Criteria for calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints_left = []  # 2D points in left image plane
imgpoints_right = []  # 2D points in right image plane

# Load images for calibration
left_images = glob.glob('left/*.jpg')  # Change the path to your left camera images
right_images = glob.glob('right/*.jpg')  # Change the path to your right camera images

# Loop through images and find chessboard corners
for left_img, right_img in zip(left_images, right_images):
    left_img = cv2.imread(left_img)
    right_img = cv2.imread(right_img)

    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        objpoints.append(objp)

        corners_left_subpix = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        imgpoints_left.append(corners_left_subpix)

        corners_right_subpix = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        imgpoints_right.append(corners_right_subpix)
    else:
        print(f"Chessboard corners not found in images: {left_img} and {right_img}")

# Print diagnostic information
print("Number of valid calibration images found:")
print(f"Left camera: {len(imgpoints_left)}")
print(f"Right camera: {len(imgpoints_right)}")

# Check if enough images were found for calibration
if len(imgpoints_left) < 1 or len(imgpoints_right) < 1:
    print("Error: Not enough valid calibration images found.")
    exit()

# Calibrate cameras
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left,
                                                                             gray_left.shape[::-1], None, None)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right,
                                                                                  gray_right.shape[::-1], None, None)

# Stereo calibration
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC  # Keep the intrinsic parameters fixed
ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right,
                                                         mtx_left, dist_left, mtx_right, dist_right,
                                                         gray_left.shape[::-1], criteria=criteria, flags=flags)

# Save calibration data
np.savez('calibration_data.npz', ret_stereo=ret_stereo, mtx_left=mtx_left, dist_left=dist_left, rvecs_left=rvecs_left,
         tvecs_left=tvecs_left, mtx_right=mtx_right, dist_right=dist_right, rvecs_right=rvecs_right,
         tvecs_right=tvecs_right, R=R, T=T, E=E, F=F)

print("Calibration completed and data saved.")
