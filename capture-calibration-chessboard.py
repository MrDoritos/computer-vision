#!/bin/python3

#from mailcap import getcaps
import numpy as np
import cv2
import importlib

getcams = importlib.import_module('getcams')

rows = 9
columns = 7

cap_left, cap_right = getcams.get_cameras()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
objp = np.zeros((rows*columns, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)

frame_left = None
frame_right = None

objpoints = []  # 3D points in real world space
imgpoints_left = []  # 2D points in image plane
imgpoints_right = []

print("Capturing chessboard data")

capture_count = 0

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    if not ret_left or not ret_right:
        break

    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (rows, columns), None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (rows, columns), None)

    if ret_left and ret_right:
        conv_size = (11, 11)
        corners_left = cv2.cornerSubPix(gray_left, corners_left, conv_size, (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, conv_size, (-1, -1), criteria)
    
        cv2.drawChessboardCorners(frame_left, (rows, columns), corners_left, ret_left)
        cv2.drawChessboardCorners(frame_right, (rows, columns), corners_right, ret_right)

        if capture_count % 10 == 0:
            print("Captured image pair", capture_count)
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)

    capture_count += 1

    # Display the frame
    cv2.imshow('Left camera', frame_left)
    cv2.imshow('Right camera', frame_right)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Calculating calibration parameters")

#print("imgpoints_left: ", imgpoints_left, "imgpoints_right: ", imgpoints_right)

def camera_calibration(camera:str, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_left.shape[:2], None, None)
    np.savez('intrinsic_chessboard_calibration_params_' + camera + '.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("camera:", camera, "rmse:", ret, "mtx:", mtx, "dist:", dist)
    return mtx, dist, rvecs, tvecs

#mtx_left, dist_left, rvecs_left, tvecs_left = camera_calibration('left', objpoints, imgpoints_left)
#mtx_right, dist_right, rvecs_right, tvecs_right = camera_calibration('right', objpoints, imgpoints_right)

mtx_left, dist_left, rvecs_left, tvecs_left = camera_calibration('left', objpoints, imgpoints_left)
mtx_right, dist_right, rvecs_right, tvecs_right = camera_calibration('right', objpoints, imgpoints_right)

print("Perform Stereo Calibration")
ret, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, frame_left.shape[:2], criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)

print('rmse:', ret, 'T:', T, 'R:', R, 'E:', E, 'F:', F)

np.savez('extrinsic_chessboard_calibration_params.npz', Kl=Kl, Dl=Dl, Kr=Kr, Dr=Dr, R=R, T=T, E=E, F=F)

R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(Kl, Dl, Kr, Dr, frame_left.shape[:2], R, T)

xmap1, ymap1 = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, frame_left.shape[:2], cv2.CV_32FC1)
xmap2, ymap2 = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, frame_left.shape[:2], cv2.CV_32FC1)

print("Displaying undistorted rectified frames")

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()


    #frame_left = cv2.undistort(frame_left, mtx_left, dist_left)
    #frame_right = cv2.undistort(frame_right, mtx_right, dist_right)
    frame_left = cv2.remap(frame_left, xmap1, ymap1, cv2.INTER_LINEAR)
    frame_right = cv2.remap(frame_right, xmap2, ymap2, cv2.INTER_LINEAR)


    cv2.imshow('Left camera', frame_left)
    cv2.imshow('Right camera', frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#np.savez('calibration_params.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Release video capture object and close windows
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
