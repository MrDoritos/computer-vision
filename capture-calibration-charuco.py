#!/bin/python3

import numpy as np
import cv2

# Load ChArUco board parameters
board = cv2.aruco.CharucoBoard([8,8], 0.019, 0.014, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
board.setLegacyPattern(True)

# Get the list of available camera devices
camera_ids = [id for id in range(10)]  # Assuming cameras are indexed from 0 to 9

# Open cameras and select the first two working cameras
cap_left = None
cap_right = None
cap_tmp = None

#cv2.VideoCapture.set(cv2.CAP_PROP_BUFFERSIZE, value=3)

for camera_id in camera_ids:
    cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        if cap_right is None:
            cap_right = cap
        elif cap_left is None:
            cap_left = cap
            break  # Found both cameras
    else:
        cap.release()

if cap_left is None or cap_right is None:
    print("Error: Could not find two working cameras.")
    exit()
else:
    print("Cameras opened")

cameras = [cap_left, cap_right]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
objp = np.zeros((8*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:8].T.reshape(-1, 2)

frame_left = None
frame_right = None

print('Switch cameras and place in front of charuco, press q to continue')
while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    cv2.imshow('Left camera', frame_left)
    cv2.imshow('Right camera', frame_right)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        cap_tmp = cap_left
        cap_left = cap_right
        cap_right = cap_tmp

objpoints = []  # 3D points in real world space
imgpoints_left = []  # 2D points in image plane
imgpoints_right = []
all_charuco_ids_left = []
all_charuco_ids_right = []

def charuco_capture(frame, board):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, board.getDictionary())
    ret = False
    charuco_ids = None

    if len(corners) > 0:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if ret > 20:
            frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
            corners = charuco_corners
            ret = True
        else:
            ret = False
            
            
    return ret, corners, charuco_ids, frame


while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    if not ret_left or not ret_right:
        break

    
    ret_left, corners_left, charuco_ids_left, frame_left = charuco_capture(frame_left, board)
    ret_right, corners_right, charuco_ids_right, frame_right = charuco_capture(frame_right, board)
    
    if ret_left and ret_right:
        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)
        all_charuco_ids_left.append(charuco_ids_left)
        all_charuco_ids_right.append(charuco_ids_right)

            #ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(charuco_corners, charuco_ids, board, gray.shape[::-1], None, None)
            #frame = cv2.undistort(frame, cameraMatrix=mtx, distCoeffs=dist)



    # Display the frame
    cv2.imshow('Left camera', frame_left)
    cv2.imshow('Right camera', frame_right)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Calculating calibration parameters")

#print("imgpoints_left: ", imgpoints_left, "imgpoints_right: ", imgpoints_right)

def camera_calibration(camera:str, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(objpoints, imgpoints, board, frame_left.shape[:2], None, None)
    np.savez('intrinsic_calibration_params_' + camera + '.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    return mtx, dist, rvecs, tvecs

#mtx_left, dist_left, rvecs_left, tvecs_left = camera_calibration('left', objpoints, imgpoints_left)
#mtx_right, dist_right, rvecs_right, tvecs_right = camera_calibration('right', objpoints, imgpoints_right)

mtx_left, dist_left, rvecs_left, tvecs_left = camera_calibration('left', imgpoints_left, all_charuco_ids_left)
mtx_right, dist_right, rvecs_right, tvecs_right = camera_calibration('right', imgpoints_right, all_charuco_ids_right)

print("Displaying undistorted frames")

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    frame_left = cv2.undistort(frame_left, mtx_left, dist_left)
    frame_right = cv2.undistort(frame_right, mtx_right, dist_right)

    cv2.imshow('Left camera', frame_left)
    cv2.imshow('Right camera', frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#np.savez('calibration_params.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()
