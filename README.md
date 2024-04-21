# VVC Computer Vision Project

## Goal

To map and store a 3D environment using a binocular camera setup. We will find software that detects points in 3D space using disparity maps. This will enable to detect relative positions of objects in an environment. We can use barcodes to make identification of reference objects easier. The intented use case is a robotic arm with vision capabilities for object recognition and object avoidance.

## Camera Information

### Camera firmware

Sensor: 1/2.7" (2.54mm/6.858mm) OV2710\n
Resolution: 2MP 1080p\n
Data Format: MJPG/YUV2\n
Frame Rate: MJPG 30fps@1080p, YUV2 30fps@480p

### Camera hardware

Lens FOV: H=100*, D=138*\n
Lens Mount: M12\n
Focusing Range: 3.3ft (1m) to infinity

### Estimated Lens Parameters

Focal Length: 10.09mm (35mm)\n
Focal Length: 1.7051mm (Actual)

### Calibrated Lens Parameters

#### Camera 1 - Left (Origin)

##### Intrinsic Parameters\n
Focal Length X: 647.49\n
Focal Length Y: 645.93\n
Optical Center X: 389.98\n
Optical Center Y: 312.70\n
##### Extrinsic Parameters\n
radial_k1: -0.393\n
radial_k2: -0.235\n
radial_k3: -0.091\n
tangential_p1: -0.002\n
tangential_p2: -0.007\n

#### Camera 2 - Right

##### Intrinsic Parameters\n
Focal Length X: 637.33\n
Focal Length Y: 636.02\n
Optical Center X: 395.87\n
Optical Center Y: 299.81\n
##### Extrinisic Parameters\n
radial_k1: -0.395\n
radial_k2: 0.319\n
radial_k3: -0.237\n
tangential_p1: -0.001\n
tangential_p2: -0.007\n
##### Position Relative to Camera 1\n
Translation: -0.0562, 0.0008, -0.0015\n
Rotation: -0.0109, 0.0447, 0.0111

## Referenced Software

Creates calibration data for the stereo camera setup\n
[caliscope](https://github.com/mprib/caliscope)\n
Recording synchronized views\n
[freemocap](https://github.com/freemocap/freemocap)\n
Referenced by caliscope, also records synchronized views, I used freemocap\n
[multiwebcam](https://github.com/mprib/multiwebcam)