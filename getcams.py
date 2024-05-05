#!/bin/python3

import cv2

def get_cameras():
    camera_ids = [id for id in range(10)]  

    cap_left = None
    cap_right = None
    cap_tmp = None

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

    print('Switch cameras and place in front of target, press q to continue')
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

    return cap_left, cap_right