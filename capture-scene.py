#!/bin/python3

import numpy as np
import cv2
import importlib
import os
import datetime

getcams = importlib.import_module("getcams")

cap_left, cap_right = getcams.get_cameras()

capture_count = 0
frame_count = 0
# Create output directory with today's date and time
output_dir = "./output/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

frame_skip = 30

path_right = output_dir + "/right"
path_left = output_dir + "/left"

os.makedirs(path_left, exist_ok=True)
os.makedirs(path_right, exist_ok=True)

# Rest of the code...

while True:
    if not (cap_left.grab() and cap_right.grab()):
        print("No more frames")
        break

    frame_count += 1

    _, frame_left = cap_left.retrieve()
    _, frame_right = cap_right.retrieve()

    cv2.imshow("Left camera", frame_left)
    cv2.imshow("Right camera", frame_right)
    
    if frame_count % frame_skip != 0:
        continue

    # Save frames
    cv2.imwrite(path_left + "/frame%d.jpg" % capture_count, frame_left)
    cv2.imwrite(path_right + "/frame%d.jpg" % capture_count, frame_right)

    capture_count += 1

    print("\rImage %d saved" % capture_count)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break