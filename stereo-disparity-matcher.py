#!/bin/python3
import cv2
import numpy as np
import os

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
# Set camera resolution
for camera in cameras:
    print("Camera properties:")
    print("FRAME_WIDTH:", camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("FRAME_HEIGHT:", camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS:", camera.get(cv2.CAP_PROP_FPS))
    print("BUFFERSIZE:", camera.get(cv2.CAP_PROP_BUFFERSIZE))
    print("AUTOFOCUS:", camera.get(cv2.CAP_PROP_AUTOFOCUS))
    print("AUTO_EXPOSURE:", camera.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    print("EXPOSURE:", camera.get(cv2.CAP_PROP_EXPOSURE))
    print("BRIGHTNESS:", camera.get(cv2.CAP_PROP_BRIGHTNESS))
    print("CONTRAST:", camera.get(cv2.CAP_PROP_CONTRAST))
    print("SATURATION:", camera.get(cv2.CAP_PROP_SATURATION))
    print("CONVERT_RGB:", camera.get(cv2.CAP_PROP_CONVERT_RGB))
    print("FORMAT:", camera.get(cv2.CAP_PROP_FORMAT))
    print("HUE:", camera.get(cv2.CAP_PROP_HUE))
    print("GAIN:", camera.get(cv2.CAP_PROP_GAIN))
    print("WHITE_BALANCE_BLUE_U:", camera.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U))
    print("WHITE_BALANCE_RED_V:", camera.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V))
    print("BUFFERSIZE:", camera.get(cv2.CAP_PROP_BUFFERSIZE))
    print("BACKEND:", camera.get(cv2.CAP_PROP_BACKEND))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    #camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    #camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    camera.set(cv2.CAP_PROP_EXPOSURE, 10.0)
    #camera.set(cv2.CAP_PROP_MODE, 0)
    #camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
    #camera.set(cv2.CAP_PROP_CONTRAST, 0.5)
    #camera.set(cv2.CAP_PROP_SATURATION, 0.5)
    #camera.set(cv2.CAP_PROP_CONVERT_RGB, 3)
    #camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    #camera.set(cv2.CAP_PROP_FORMAT, cv2.8UC1)
    #camera.set(cv2.CAP_PROP_HUE, 0.5)
    #camera.set(cv2.CAP_PROP_GAIN, 0.5)
    #camera.set(cv2.CAP_PROP_SETTINGS, 0)

frame_count = 0  # Counter for captured frames
image_count = 0
   
window_size = 3
min_disp = 16
num_disp = 112-min_disp
stereo = cv2.StereoSGBM.create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    mode=cv2.StereoSGBM_MODE_HH
)

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Error: Failed to capture frame from one or both cameras.")
        break

    # Display frames (optional)
    cv2.imshow('Left Camera', frame_left)
    cv2.imshow('Right Camera', frame_right)

    disp = stereo.compute(frame_left, frame_right).astype(np.float32) / 16.0
    h, w = frame_left.shape[:2]
    f = 0.8*w                          # guess for focal length

    cv2.imshow('Disparity', (disp-min_disp)/num_disp)

    key = cv2.waitKey(1) & 0xFF
    # Exit loop if 'q' is pressed
    if key == ord('q'):
        break

    # Switch cameras
    if key == ord('s'):
        cap_tmp = cap_left
        cap_left = cap_right
        cap_right = cap_tmp
        print("Cameras switched")

# Release cameras and close OpenCV windows
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

print(f"{frame_count} frames captured and saved.")
