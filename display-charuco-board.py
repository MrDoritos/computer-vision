#!/bin/python3

import numpy as np
import cv2

# Load ChArUco board parameters
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(7, 5, 0.04, 0.02, dictionary)

# Create video capture object
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers and corners
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary)
    if len(corners) > 0:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if ret > 20:
            frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

    # Display the frame
    cv2.imshow('ChArUco Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()
