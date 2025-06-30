#!/bin/python3
import numpy as np
import cv2, glob, os, sys, json, math, threading, datetime, time
from threading import Lock, Condition, RLock, Thread
from cv2.typing import MatLike

class Util:
    def diagonal(width, height):
        return math.sqrt(width ** 2 + height ** 2)

class Intrinsics:
    diagonal_35mm = Util.diagonal(36, 24)
    focal_length_35mm = 35

    def crop_factor(diagonal_mm):
        return Intrinsics.diagonal_35mm / diagonal_mm
    
    def effective_focal_length(focal_length, crop_factor):
        return focal_length / crop_factor
    
    def focal_length(effective_focal_length, crop_factor):
        return effective_focal_length * crop_factor
    
    def relative_focal_length(focal_length):
        return focal_length / Intrinsics.focal_length_35mm
    
    def absolute_focal_length(focal_length_relative):
        return focal_length_relative * Intrinsics.focal_length_35mm

class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get_area(self):
        return self.width * self.height
    
    def get_diagonal(self):
        return Util.diagonal(self.width, self.height)
    
    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def get_length(self):
        return self.height if self.height > self.width else self.width
    
    def get_aspect_ratio(self):
        return self.width / self.height
    
    def get_size(self):
        return Size(self.width, self.height)

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def set_size(self, width, height):
        self.set_width(width)
        self.set_height(height)

    def set_diagonal(self, diagonal):
        self.scale(diagonal / self.get_diagonal())

    def scale(self, v):
        self.set_size(self.width * v, self.height * v)

    def to_scale(self, v):
        return Size(self.width * v, self.height * v)
    
    def __repr__(self, pretty:bool=False):
        s = f'[width: {self.width}, height: {self.height}, aspect_ratio: {self.get_aspect_ratio()}, area: {self.get_area()}, diagonal: {self.get_diagonal()}, length: {self.get_length()}]'
        if pretty:
            return s + '\n'
        return s

class Image(Size):
    def __init__(self, width, height):
        Size.__init__(self, width, height)

    def get_megapixels(self):
        return self.get_area() * 0.000001
    
    def set_megapixels(self, megapixels):
        self.scale(megapixels / self.get_megapixels())

    def __repr__(self, pretty:bool=False):
        s = f'[image_size: {self.get_size().__repr__(pretty)}, megapixels: {self.get_megapixels()}MP]'
        if pretty:
            return s + '\n'
        return s

class Sensor(Size):
    def __init__(self, width, height, pixel_size):
        Size.__init__(self, width, height)
        self.pixel_size = pixel_size

    def get_crop_factor(self):
        return Intrinsics.crop_factor(self.get_diagonal())
    
    def get_pixel_size(self):
        return self.pixel_size
    
    def get_pixel_size_um(self):
        return self.get_pixel_size() * 1000
    
    def get_image_size(self):
        return Image(self.get_width()/self.get_pixel_size(),self.get_height()/self.get_pixel_size())
    
    def set_pixel_size(self, pixel_size):
        self.pixel_size = pixel_size

    def set_pixel_size_um(self, micrometers):
        self.set_pixel_size(micrometers * 0.001)

    def set_crop_factor(self, crop_factor):
        self.set_pixel_size(self.get_pixel_size() * (crop_factor / self.get_crop_factor()))

    def __repr__(self, pretty:bool=False):
        s = f'[pixel_size: {self.get_pixel_size()}mm, crop_factor: {self.get_crop_factor()}, sensor_size: {self.get_size().__repr__(pretty)}, image_size: {self.get_image_size().__repr__(pretty)}]'
        if pretty:
            return s + '\n'
        return s

class Lens:
    def __init__(self, sensor:Sensor, image:Image, focal_length:float):
        self.sensor = sensor
        self.image = image
        self.focal_length = focal_length

    def get_effective_focal_length(self):
        return Intrinsics.effective_focal_length(self.get_focal_length(), self.sensor.get_crop_factor())
    
    def get_focal_length(self):
        return self.focal_length
    
    def get_relative_focal_length(self):
        return Intrinsics.relative_focal_length(self.get_focal_length())
    
    def set_focal_length(self, focal_length):
        self.focal_length = focal_length

    def set_effective_focal_length(self, effective_focal_length):
        self.set_focal_length(Intrinsics.focal_length(effective_focal_length, self.sensor.get_crop_factor()))

    def __repr__(self, pretty:bool=False):
        s = f'[sensor: {self.sensor.__repr__(pretty)}, image: {self.image.__repr__(pretty)}, F35: {self.get_focal_length()}mm, Fefl: {self.get_effective_focal_length()}mm, Frel: {self.get_relative_focal_length()}]'
        if pretty:
            return s + '\n'
        return s

class Device:
    props={
        'fps': cv2.CAP_PROP_FPS,
        'width': cv2.CAP_PROP_FRAME_WIDTH,
        'height': cv2.CAP_PROP_FRAME_HEIGHT,
        'fourcc': cv2.CAP_PROP_FOURCC,
        'buffersize': cv2.CAP_PROP_BUFFERSIZE,
    }

    def __init__(self, camera_id_or_path, name:str=None, auto_open:bool=True):
        self.device = None
        self.size = None
        self.path = camera_id_or_path
        self.name = name if name is not None else str(self.path)

        if self.path is not None and auto_open:
            self.open(camera_id_or_path)

    def __del__(self):
        self.close()

    def flush(self):
        while True:
            ret, _ = self.device.read()
            if not ret:
                break

    def grab(self) -> bool:
        return self.device.grab()

    def read(self):
        ret,frame = self.device.read()
        if not ret:
            return None
        return frame
    
    def retrieve(self):
        ret, frame = self.device.retrieve()
        if not ret:
            return None
        return frame
    
    def get_or_set_path(self, path=None):
        if path is None:
            return self.path
        self.path = path
        return path        

    def close(self):
        if self.is_open():
            self.device.release()
            print(f'Device {self.get_name()} closed')

    def _device_open(self):
        self.device = cv2.VideoCapture(self.get_path())

    def open(self, camera_id_or_path=None):
        if self.is_open():
            self.close()
        
        if self.get_or_set_path(camera_id_or_path) is not None:
            self._device_open()

        if self.is_open():
            print(f'Device {self.get_name()} opened')

    def has_device(self):
        return self.device is not None

    def is_open(self):
        return self.has_device() and self.device.isOpened()


    def get_key(self, key):
        if isinstance(key, int):
            return key
        if key in self.props:
            return self.props[key]
        raise ValueError("Parameter key not int or string")

    def get_parameter(self, key):
        return self.device.get(self.get_key(key))

    def set_parameter(self, key, value):
        self.device.set(self.get_key(key), value)

    def set_parameters(self, **kwargs):
        for key, value in kwargs:
            self.set_parameter(key, value)


    def get_path(self):
        return self.path

    def get_name(self):
        return self.name

    def get_fps(self):
        return self.get_parameter('fps')
    
    def get_frametime(self):
        return 1/self.get_fps()

    def get_width(self):
        return self.get_parameter('width')

    def get_height(self):
        return self.get_parameter('height')

    def get_size(self):
        return Size(self.get_width(), self.get_height())


    def set_path(self, path):
        self.open(path)

    def set_name(self, name):
        self.name = name

    def set_fps(self, fps:float=30):
        self.set_parameter('fps', fps)

    def set_frametime(self, frametime:float):
        self.set_fps(1/frametime)

    def set_width(self, width):
        self.set_parameter('width', width)

    def set_height(self, height):
        self.set_parameter('height', height)

    def set_size(self, size:Size):
        self.set_width(size.get_width())
        self.set_height(size.get_height())

class DeviceThread:
    def __init__(self, device:Device, condition:Condition, buffer_size:int=1):
        self.device = device
        self.running = False
        self.thread = None
        self.condition = condition
        self.loop_count = 0
        self.frame_count = 0
        self.buffer_size = buffer_size
        self.frames = []

        self.data_lock = Lock()
        self.working_lock = Lock()

    def push_frame(self, frame):
        with self.data_lock:
            self.frames.append(frame)
            count = len(self.frames)
            if count > self.buffer_size:
                self.frames = self.frames[-self.buffer_size:]

    def pop_frame(self) -> MatLike:
        with self.data_lock:
            return self.frames.pop(0)
    
    def try_pop_frame(self) -> None|MatLike:
        with self.data_lock:
            count = len(self.frames)
            if count < 1:
                return None
            return self.frames.pop(0)

    def get_framebuffer_length(self) -> int:
        with self.data_lock:
            return len(self.frames)
    
    def has_frame(self) -> bool:
        return self.get_framebuffer_length() > 0
    
    def is_running(self) -> bool:
        return self.running
    
    def is_thread_active(self) -> bool:
        return self.thread is not None and self.thread.is_alive()
    
    def is_thread_dead(self) -> bool:
        return self.thread is not None and not self.thread.is_alive()

    def is_normal(self) -> bool:
        return self.is_running() and self.is_thread_active()

    def is_waiting(self) -> bool:
        return self.working_lock.locked()

    def should_run(self) -> bool:
        return self.is_running() and self.device.is_open()

    def thread_wait(self):
        with self.condition:
            self.working_lock.release()
            self.condition.wait()
        self.working_lock.acquire(True)

    def wait(self, timeout=10):
        if self.working_lock.acquire(True, timeout):
            self.working_lock.release()

    def device_thread(self):
        self.working_lock.acquire(True)

        try:
            while self.should_run():
                if self.device.grab():
                    frame = self.device.retrieve()
                    if frame is not None:
                        self.push_frame(frame)
                        self.frame_count += 1

                self.loop_count += 1

                self.thread_wait()
        except:
            print('Device thread threw exception')
        finally:
            if self.working_lock.locked():
                self.working_lock.release()

    def start(self):
        if not self.is_running():
            self.running = True

        if self.is_thread_active():
            return

        self.thread = Thread(target=self.device_thread)
        self.thread.start()

    def notify(self):
        with self.condition:
            self.condition.notify_all()

    def join(self):
        if not self.thread or self.is_running():
            return
        
        self.wait()
        self.notify()

        if self.thread.is_alive():
            self.thread.join()

        self.thread = None

    def stop(self):
        if self.is_running():
            self.running = False

        if self.is_thread_dead():
            self.thread = None
            return    
    
        self.join()


class Capture:
    def __init__(self, devices:list[Device]=None, camera_count:int=10, fps:int=30, auto_find_devices:bool=True):
        self.fps = fps
        self.frame_count = 0
        self.start_time = datetime.datetime.now()
        self.last_frame_time = None
        self.devices = devices
        
        if (self.devices is None or len(self.devices) < 1) and auto_find_devices:
            self.search_devices(camera_count)

    def set_parameters(self, **kwargs):
        for device in self.devices:
            device.set_parameters(kwargs)

    def get_device(self, key:str) -> Device:
        for device in self.devices:
            if device.name == key:
                return device

    def open_devices(self):
        for device in self.devices:
            if not device.is_open():
                device.open()

    def close_devices(self):
        for device in self.devices:
            device.close()

    def get_open_devices(self) -> list[Device]:
        devices=[]
        for device in self.devices:
            if device.is_open():
                devices.append(device)
        return devices
    
    def tick_frame(self):
        self.frame_count += 1

    def get_fps(self) -> float:
        return self.fps

    def get_frametime(self) -> float:
        return 1/self.get_fps()

    def get_wait_timedelta(self, tp=None) -> datetime.timedelta:
        if self.last_frame_time is None:
            return datetime.timedelta(seconds=0)
        if tp is None:
            tp = datetime.datetime.now()
        ft=datetime.timedelta(seconds=self.get_frametime())
        return self.last_frame_time - (tp - ft)

    def get_wait_time(self, tp=None) -> float:
        seconds = self.get_wait_timedelta(tp).total_seconds()
        if seconds > 0:
            return seconds
        return 0

    def tick_time(self) -> float:
        now = datetime.datetime.now()
        delta = self.get_wait_timedelta(now)
        seconds = delta.total_seconds()
        self.last_frame_time = now
        if seconds > 0:
            self.last_frame_time += delta
            return seconds
        return 0

    def wait(self):
        """
        Waits for the next frame to be ready based on the capture fps

        Modifies the last_frame_time to be now + remaining 
            time until the next frame should be captured

        Effectively last_frame_time will be the time when the function returns
        """
        seconds = self.tick_time()
        if seconds > 0:
            time.sleep(seconds)

    def get_frames(self, skip_none:bool=True) -> list[tuple[Device, MatLike]]:
        frames=[]
        for device in self.devices:
            device.grab()
        for device in self.devices:
            frame = device.retrieve()
            if frame is not None or not skip_none:
                frames.append((device, frame))
        self.tick_frame()
        return frames

    def flush_devices(self):
        for device in self.devices:
            device.flush()

    def remove_closed_devices(self):
        self.devices = self.get_open_devices()

    def search_devices(self, max_device_count:int=10, max_device_search_id:int=10):
        self.devices = Capture.find_devices(max_device_count, max_device_search_id)

    def find_devices(max_device_count:int=10, max_device_search_id:int=10) -> list[Device]:
        devices = []
        for id in range(max_device_search_id + 1):
            dev = Device(id)

            if dev.is_open():
                devices.append(dev)

            if len(devices) >= max_device_count:
                break
        return devices
    
class CaptureThread:
    def __init__(self, capture:Capture=None, remove_closed:bool=True, stop_on_no_devices:bool=True, auto_start_device_threads:bool=True):
        self.capture = capture if capture is not None else Capture()
        self.device_threads = []
        self.condition = Condition()
        self.running = False
        self.thread = None
        self.thread_condition = Condition()
        self.out_condition = None
        self.remove_closed = remove_closed
        self.stop_on_no_devices = stop_on_no_devices

        if auto_start_device_threads:
            self.start_devices()

    def __del__(self):
        self.stop()

    def stop(self):
        self.stop_thread()
        self.stop_devices()

    def wait(self, timeout:int=10):
        for thread in self.device_threads:
            thread.wait(timeout)

    def get_worker_count(self) -> int:
        return len(self.device_threads)

    def has_workers(self) -> bool:
        return self.get_worker_count() > 0

    def notify(self):
        with self.condition:
            self.condition.notify_all()

    def start_devices(self):
        if self.has_workers():
            self.stop_devices()

        self.device_threads = self.make_device_threads()

        for thread in self.device_threads:
            thread.start()
    
    def make_device_threads(self) -> list[DeviceThread]:
        device_threads = []
        for device in self.capture.devices:
            device_threads.append(DeviceThread(device, self.condition))
        return device_threads

    def get_frames(self, skip_none:bool=True) -> list[tuple[Device, MatLike]]:
        frames=[]
        for device in self.device_threads:
            frame = device.try_pop_frame()
            if frame is not None or not skip_none:
                frames.append((device.device,frame))
        self.capture.tick_frame()
        return frames

    def stop_devices(self):
        for thread in self.device_threads:
            thread.stop()

    def loop(self, callback=None):
        """
        When callback is used, it should be on the main thread 
            if you wish to use imshow

        Callback will be called with the ready frames

        If callback returns Truthy, the loop will end
        """
        while self.should_not_quit():
            self.notify()
            self.wait()
            if callback is not None:
                if callback(self.get_frames()):
                    break
            self.capture.remove_closed_devices()
            self.capture.wait()

    def thread_loop(self):
        while self.should_run():
            self.notify()
            self.wait()
            self.notify_out()
            self.capture.remove_closed_devices()
            self.cv_capture_wait()

    def notify_out(self):
        with self.out_condition:
            self.out_condition.notify_all()

    def cv_capture_wait(self):
        with self.thread_condition:
            timeout = self.capture.tick_time()
            self.thread_condition.wait(timeout)

    def is_running(self) -> bool:
        return self.running
    
    def should_not_quit(self) -> bool:
        return not self.stop_on_no_devices or self.has_workers()

    def should_run(self) -> bool:
        return self.is_running() and self.should_not_quit()

    def is_thread_active(self) -> bool:
        return self.thread is not None and self.thread.is_alive()
    
    def is_thread_dead(self) -> bool:
        return self.thread is not None and not self.thread.is_alive()

    def notify_thread(self):
        with self.thread_condition:
            self.thread_condition.notify_all()

    def join(self):
        if not self.thread:
            return
        
        self.notify_thread()
        
        if self.thread.is_alive():
            self.thread.join()

        self.thread = None

    def start_thread(self, condition:Condition):
        if not self.is_running():
            self.running = True

        if self.is_thread_active():
            return

        self.out_condition = condition        
        self.thread = Thread(target=self.thread_loop)
        self.thread.start()

    def stop_thread(self):
        if self.is_running():
            self.running = False

        if self.is_thread_dead():
            self.thread = None
            return
        
        self.join()
        
class CapturePreview:
    def __init__(self, capture:Capture=None, auto_start:bool=True):
        self.capture_thread = CaptureThread(capture)
        if auto_start:
            self.start()

    def start(self):
        try:
            self.capture_thread.loop(self.callback)
        except:
            print('CaptureThread loop threw exception')

        self.capture_thread.stop()
        cv2.destroyAllWindows()

    def callback(self, frames):
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return True
        
        for device, frame in frames:
            cv2.imshow(device.get_name(), frame)