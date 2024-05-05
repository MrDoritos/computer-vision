#!/bin/python3

import numpy as np
import cv2

# Load ChArUco board parameters
board = cv2.aruco.CharucoBoard([8,8], 0.019, 0.014, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
board.setLegacyPattern(True)

# Create video capture object
cap = cv2.VideoCapture(0)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers and corners
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, board.getDictionary())
    if len(corners) > 0:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

        if ret > 5:
            frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
            
            ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(charuco_corners, charuco_ids, board, gray.shape[::-1], None, None)
            frame = cv2.undistort(frame, cameraMatrix=mtx, distCoeffs=dist)



    # Display the frame
    cv2.imshow('ChArUco Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez('calibration_params.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()
