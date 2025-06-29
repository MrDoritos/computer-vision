#!/bin/python3
import cv2
import os
import threading as thread
from threading import Condition, Lock, RLock
import signal
import time
import datetime

frametime=0.1
run_threads=True
cond_getframe=Condition(RLock())
lock_readyframe=[]
threads=[]
framelist_lock=Lock()
frames={}
made_dirs=[]

output_dir = "./output/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def frame_count_to_seconds(frame_count):
    return frame_count * frametime

def save_frames(frames, frame_count, do_message=True):
    global made_dirs
    sec_file = f'frame_{frame_count}_{frame_count_to_seconds(frame_count):.3f}s.jpg'
    for i,frame in frames.items():
        cam_dir = os.path.join(output_dir, f'{i}')
        if cam_dir not in made_dirs:
            os.makedirs(cam_dir, exist_ok=True)
            made_dirs.append(cam_dir)
        cam_path = os.path.join(cam_dir, sec_file)
        cv2.imwrite(cam_path, frame)
        if do_message:
            print(f'Saved {cam_path}')

def set_device_parameters(device):
    device.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    device.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    device.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    device.set(cv2.CAP_PROP_FPS, int((1/frametime)//1))
    device.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

def find_devices(max_id=2):
    devices = []
    for id in range(max_id+1):
        cap = cv2.VideoCapture(id)
        if cap.isOpened():
            set_device_parameters(cap)
            devices.append(cap)
            continue
        cap.release()
    return devices

def camera_output_thread(camera, ready_lock, camera_name='camera'):
    global cond_getframe, frames, framelist_lock
    
    frame_count = 0
    ready_lock.acquire(True)

    while run_threads and camera.isOpened():
        ret,frame=camera.read()

        if ret:
            with framelist_lock:
                frames[camera_name] = frame

        frame_count += 1

        cond_getframe.acquire(False)
        ready_lock.release()
        cond_getframe.wait()
        ready_lock.acquire(True)
        cond_getframe.release()

    camera.release()

def verify_threads_done(timeout=10):
    for lock in lock_readyframe:
        lock.acquire(True, timeout)
        lock.release()

def handle_stop(signum=None, frame=None):
    global run_threads, cond_getframe, threads
    verify_threads_done(1)
    run_threads = False
    cond_getframe.acquire(False)
    cond_getframe.notify_all()
    cond_getframe.release()
    print("Stopping...")
    for thread in threads:
        if thread.is_alive():
            thread.join()
    cv2.destroyAllWindows()
    print("Done.")
    exit(0)

if __name__ == "__main__":
    devices=find_devices()
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGQUIT, handle_stop)
    for i,device in enumerate(devices):
        lock=Lock()
        lock_readyframe.append(lock)
        worker=thread.Thread(target=camera_output_thread, args=(device, lock, i))
        worker.start()
        threads.append(worker)

    frame_count = 0
    while True:
        cond_getframe.acquire(False)
        cond_getframe.notify_all()
        cond_getframe.release()

        verify_threads_done()

        framelist_lock.acquire(True)
        for camera_name,frame in frames.items():
            cv2.imshow(f"camera {camera_name}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            save_frames(frames, frame_count)

        frames.clear()
        framelist_lock.release()

        if key == ord('q'):
            break

        frame_count += 1
        time.sleep(frametime)
    handle_stop()