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

class ArucoHelper:
    def __init__(self, width=11, height=13, cell_width=0.015, marker_width=0.011, aruco_dict:int=cv2.aruco.DICT_4X4_100, min_corners:int=10):
        self.width = width
        self.height = height
        self.cell_width = cell_width
        self.marker_width = marker_width
        self.aruco_dict = aruco_dict
        self.min_corners = min_corners
        self.board = self.get_board()
        self.objp = self.get_objp()
        self.dictionary = self.get_dictionary()
        self.criteria = self.get_criteria()

    def get_objp(self):
        objp = np.zeros((self.width * self.height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.width, 0:self.height].T.reshape(-1, 2)
        return objp
    
    def get_board(self):
        board = cv2.aruco.CharucoBoard([self.width, self.height], self.cell_width, self.marker_width, cv2.aruco.getPredefinedDictionary(self.aruco_dict))
        board.setLegacyPattern(True)
        return board
    
    def get_dictionary(self):
        return self.board.getDictionary()
    
    def get_criteria(self):
        return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
class ArucoImage:
    def __init__(self, image_path):
        self.image_path = image_path
        self.gray = None
        self.shape = None
        self.image = None
        self.corners = None
        self.ids = None
        self.charuco_corners = None
        self.charuco_ids = None

    def load(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            return False
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.shape = self.gray.shape[::-1]
        return True

    def detect_markers(self, aruco:ArucoHelper):
        corners, ids, _ = cv2.aruco.detectMarkers(self.gray, aruco.dictionary)
        if corners is None or ids is None:
            return False
        self.corners = corners
        self.ids = ids
        return True
    
    def is_loaded(self):
        return (self.shape is not None and 
                self.gray is not None and
                self.image is not None)

    def is_markers_detected(self):
        return (self.corners is not None and
                self.ids is not None)

    def is_corners_interpolated(self):
        return (self.charuco_corners is not None and
                self.charuco_ids is not None)

    def is_complete(self):
        return (self.is_loaded() and 
                self.is_markers_detected() and 
                self.is_corners_interpolated())

    def interpolate_corners(self, aruco:ArucoHelper):
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(self.corners, self.ids, self.gray, aruco.board)
        if ret < aruco.min_corners:
            return False
        self.charuco_corners = charuco_corners
        self.charuco_ids = charuco_ids
        return True
    
    def process(self, aruco:ArucoHelper):
        if not self.is_loaded():
            if not self.load():
                return False
        
        if not self.is_markers_detected():
            if not self.detect_markers(aruco):
                return False
        
        if not self.is_corners_interpolated():
            if not self.interpolate_corners(aruco):
                return False
        
        return True

    def get_error(self):
        if not self.is_loaded():
            return "Image could not be loaded"
        elif not self.is_markers_detected():
            return "Markers could not be detected"
        elif not self.is_corners_interpolated():
            return "Corners could not be interpolated"
        return "No error"

class ArucoCamera:
    def __init__(self, camera_name, camera_path):
        self.camera_name = camera_name
        self.camera_path = camera_path
        self.object_points = None
        self.charuco_corners = None
        self.charuco_ids = None
        self.shape = None
        self.intrinsics = None
        self.failed_image_indicies=None
        self.images = self.get_images()

    def get_images(self):
        globbed = glob.glob('*.jpg', root_dir=self.camera_path)
        if not len(globbed):
            return []
        images = []
        for image in globbed:
            if '_undistort' in image:
                continue
            image_path = os.path.join(self.camera_path, image)
            images.append(ArucoImage(image_path))
        return images
    
    def charuco_points(self, aruco:ArucoHelper):
        self.object_points = []
        self.charuco_corners = []
        self.charuco_ids = []
        self.failed_image_indicies = []
        self.shape = None

        for i,image in enumerate(self.images):
            if not image.process(aruco):
                print(f'{image.image_path} Error: {image.get_error()}')
                self.failed_image_indicies.append(i)
                continue
            print(f'Processed image {image.image_path}')

            self.object_points.append(aruco.objp)
            self.charuco_corners.append(image.charuco_corners)
            self.charuco_ids.append(image.charuco_ids)
            self.shape = image.shape

        if len(self.charuco_corners) < 1:
            return False
        
        return True

    def save_json(self, path=None):
        if path is None:
            path = self.camera_path + '_calibration.json'

        jobj = {
            'ret': self.intrinsics['ret'],
            'mtx': self.intrinsics['mtx'].tolist(),
            'dist': self.intrinsics['dist'].tolist(),
        }

        with open(path, 'w') as file:
            json.dump(jobj, file)

        print(f'Saved {path}')

    def save_npz(self, path=None):
        if path is None:
            path = self.camera_path + '_calibration.npz'
        
        np.savez(path, 
                 ret=self.intrinsics['ret'],
                 mtx=self.intrinsics['mtx'],
                 dist=self.intrinsics['dist'],
                 rvecs=self.intrinsics['rvecs'],
                 tvecs=self.intrinsics['tvecs'])

        print(f'Saved {path}')

    def get_intrinsics(self, pixel_size):
        image = Image(self.shape[0], self.shape[1])
        ss=image.to_scale(pixel_size)
        sensor = Sensor(ss.width, ss.height, pixel_size)
        ins = self.intrinsics
        mtx = ins['mtx']
        fx=mtx[0][0]
        fy=mtx[1][1]
        px=mtx[0][2]
        py=mtx[1][2]
        length=image.get_length()
        fxr=length/fx
        fyr=length/fy
        efl=Util.diagonal(fxr,fyr)*math.sqrt(2)
        f35=Intrinsics.focal_length(efl, sensor.get_crop_factor())
        return Lens(sensor, image, f35)

    def save_undistort_preview(self, path=None, image:ArucoImage=None):
        if image is None:
            image = self.images[0]

        if path is None:
            path = self.camera_path + f'_{os.path.basename(image.image_path)}_undistort.jpg'

        mtx,dist=self.intrinsics['mtx'],self.intrinsics['dist']
        cv2.imwrite(path, cv2.undistort(self.images[0].image, mtx, dist))

        print(f'Saved {path}')

    def save(self):
        self.save_json()
        self.save_npz()
        self.save_undistort_preview()        

    def process(self, aruco:ArucoHelper):
        if not self.has_images():
            return False
        
        if not self.has_charuco_points():
            if not self.charuco_points(aruco):
                return False
        
        if not self.has_calibration():
            if not self.calibrate(aruco):
                return False
            
        return True

    def has_charuco_points(self):
        return (self.charuco_corners is not None and 
                len(self.charuco_corners) > 0)
    
    def has_images(self):
        return (self.images is not None and
                len(self.images) > 0)
    
    def has_calibration(self):
        return (self.intrinsics is not None)
    
    def is_complete(self):
        return (self.has_images() and
                self.has_charuco_points() and
                self.has_calibration())

    def get_valid_images(self):
        ret=[]
        for image in self.images:
            if image.is_complete():
                ret.append(image)
        return ret

    def calibrate(self, aruco:ArucoHelper):
        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            self.charuco_corners, 
            self.charuco_ids, 
            aruco.board, 
            self.shape, 
            None, 
            None
        )

        self.intrinsics = {
            'ret':ret,
            'mtx':mtx,
            'dist':dist,
            'rvecs':rvecs,
            'tvecs':tvecs,
        }

        return True
    
    def get_error(self):
        if not self.has_images():
            return "Camera has no images"
        elif not self.has_charuco_points():
            return "Charuco points could not be found"
        elif not self.has_calibration():
            return "Calibration could not be calculated"
        return "No error"

class ArucoRig:
    def __init__(self, rig_path):
        self.rig_path = rig_path
        self.cameras = []
        self.extrinsics = None
        self.find_cameras(self.rig_path)

    def has_cameras(self):
        return (self.cameras is not None and 
                len(self.cameras) > 0)

    def find_cameras(self, path):
        dirname = os.path.basename(path)
        for relpath, dirs, _ in os.walk(path):
            for dir in dirs:
                dirpath=os.path.join(relpath, dir)
                self.find_cameras(dirpath)
        cam = ArucoCamera(dirname, path)
        if cam.has_images():
            self.cameras.append(cam)

    def get_valid_cameras(self):
        ret = []
        for camera in self.cameras:
            if camera.is_complete():
                ret.append(camera)
        return ret
    
    def get_valid_camera_count(self):
        return len(self.get_valid_cameras())

    def save_extrinsics(self):
        json_path = os.path.join(self.rig_path, 'rig_extrinsics.json')
        npz_path = os.path.join(self.rig_path, 'rig_extrinsics.npz')

        ext=self.extrinsics

        ret,Kl,Dl,Kr,Dr,R,T,E,F=ext.values()

        jobj = {
            'ret':ext['ret'],
            'Kl':ext['Kl'].tolist(),
            'Dl':ext['Dl'].tolist(),
            'Kr':ext['Kr'].tolist(),
            'Dr':ext['Dr'].tolist(),
            'R':ext['R'].tolist(),
            'T':ext['T'].tolist(),
            'E':ext['E'].tolist(),
            'F':ext['F'].tolist(),
        }

        with open(json_path, 'w') as file:
            json.dump(jobj, file)

        print(f'Saved {json_path}')

        np.savez(npz_path, 
                 ret=ret,
                 Kl=Kl,
                 Dl=Dl,
                 Kr=Kr,
                 Dr=Dr,
                 R=R,
                 T=T,
                 E=E,
                 F=F)
        
        print(f'Saved {npz_path}')

    def save(self):
        for camera in self.get_valid_cameras():
            camera.save()

        if self.has_extrinsics():
            self.save_extrinsics()

    def has_calibrations(self):
        return self.get_valid_camera_count() > 0
    
    def has_extrinsics(self):
        return self.extrinsics is not None

    def is_complete(self):
        return self.has_calibrations()

    def calibrations(self, aruco:ArucoHelper):
        for camera in self.cameras:
            if not camera.process(aruco):
                print(f'{camera.camera_name} Error: {camera.get_error()}')
                continue
            print(f'Processed camera {camera.camera_name}')

    def get_error(self):
        if not self.has_cameras():
            return "No valid cameras"
        elif not self.has_calibrations():
            return "No calibrated cameras"
        return "No error"

    def stereo_calibrate(self, aruco:ArucoHelper, cam_a:ArucoCamera, cam_b:ArucoCamera):
        mtx_a,mtx_b=cam_a.intrinsics['mtx'],cam_b.intrinsics['mtx']
        dist_a,dist_b=cam_a.intrinsics['dist'],cam_b.intrinsics['dist']

        fail_a,fail_b = cam_a.failed_image_indicies,cam_b.failed_image_indicies
        img_a,img_b = cam_a.images,cam_b.images
        imgc_a,imgc_b = len(img_a),len(img_b)
        imgc_min = min(imgc_a, imgc_b)
        allfail=fail_a+fail_b

        objpoints,imgpoints_a,imgpoints_b=[],[],[]

        for i in range(imgc_min):
            if i in allfail:
                continue
            ccs_a, cids_a=cam_a.images[i].charuco_corners,cam_a.images[i].charuco_ids
            ccs_b, cids_b=cam_b.images[i].charuco_corners,cam_b.images[i].charuco_ids
            op_a,ip_a=cv2.aruco.getBoardObjectAndImagePoints(aruco.board, ccs_a, cids_a)
            op_b,ip_b=cv2.aruco.getBoardObjectAndImagePoints(aruco.board, ccs_b, cids_b)
            if len(op_a) != len(op_b):
                continue
            objpoints.append(op_a)
            imgpoints_a.append(ip_a)
            imgpoints_b.append(ip_b)

        ret, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_a, imgpoints_b,
            mtx_a, dist_a, mtx_b, dist_b,
            cam_a.shape, criteria=aruco.criteria, flags=cv2.CALIB_FIX_INTRINSIC
        )

        self.extrinsics = {
            'ret': ret,
            'Kl': Kl,
            'Dl': Dl,
            'Kr': Kr,
            'Dr': Dr,
            'R': R,
            'T': T,
            'E': E,
            'F': F,
        }

        return True

    def process(self, aruco:ArucoHelper):
        if not self.has_cameras():
            return False
        if not self.has_calibrations():
            self.calibrations(aruco)
        if not self.has_calibrations():
            return False
        if not self.has_extrinsics() and self.get_valid_camera_count() > 1:
            print('Valid stereo rig detected')
            cam_a,cam_b=self.get_valid_cameras()[:2]
            if not self.stereo_calibrate(aruco, cam_a, cam_b):
                return False
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Give path to rig or images")
    
    aruco = ArucoHelper()
    rig = ArucoRig(sys.argv[1])

    if not rig.process(aruco):
        print(f'Rig Error: {rig.get_error()}')

    rig.save()

    for camera in rig.get_valid_cameras():
        print(camera.camera_path, camera.get_intrinsics(0.003).__repr__(True))

    print('Done')