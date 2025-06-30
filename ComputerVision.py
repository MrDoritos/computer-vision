#!/bin/python3
import numpy as np
import cv2, glob, os, sys, json, math

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
