# VVC Computer Vision Project

## Goal

To map and store a 3D environment using a binocular camera setup. We will find software that detects points in 3D space using disparity maps. This will enable to detect relative positions of objects in an environment. We can use barcodes to make identification of reference objects easier. The intented use case is a robotic arm with vision capabilities for object recognition and object avoidance.

## Camera Information

### Camera firmware

Sensor: 1/2.7" (2.54mm/6.858mm) OV2710<br />
Resolution: 2MP 1080p<br />
Data Format: MJPG/YUV2<br />
Frame Rate: MJPG 30fps@1080p, YUV2 30fps@480p

### Camera hardware

Lens FOV: H=100*, D=138*<br />
Lens Mount: M12<br />
Focusing Range: 3.3ft (1m) to infinity

### Estimated Lens Parameters

Focal Length: 10.09mm (35mm)<br />
Focal Length: 1.7051mm (Actual)

### Calibrated Lens Parameters

#### Camera 1 - Left (Origin)

##### Intrinsic Parameters<br />
Focal Length X: 647.49<br />
Focal Length Y: 645.93<br />
Optical Center X: 389.98<br />
Optical Center Y: 312.70<br />
##### Extrinsic Parameters<br />
radial_k1: -0.393<br />
radial_k2: -0.235<br />
radial_k3: -0.091<br />
tangential_p1: -0.002<br />
tangential_p2: -0.007<br />

#### Camera 2 - Right

##### Intrinsic Parameters<br />
Focal Length X: 637.33<br />
Focal Length Y: 636.02<br />
Optical Center X: 395.87<br />
Optical Center Y: 299.81<br />
##### Extrinisic Parameters<br />
radial_k1: -0.395<br />
radial_k2: 0.319<br />
radial_k3: -0.237<br />
tangential_p1: -0.001<br />
tangential_p2: -0.007<br />
##### Position Relative to Camera 1<br />
Translation: -0.0562, 0.0008, -0.0015<br />
Rotation: -0.0109, 0.0447, 0.0111

## Referenced Software

Creates calibration data for the stereo camera setup<br />
[caliscope](https://github.com/mprib/caliscope)<br />
Recording synchronized views<br />
[freemocap](https://github.com/freemocap/freemocap)<br />
Referenced by caliscope, also records synchronized views, I used freemocap<br />
[multiwebcam](https://github.com/mprib/multiwebcam)
