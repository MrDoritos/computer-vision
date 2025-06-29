#!/bin/python3
import numpy as np
import cv2, glob, os, sys, json

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
        self.shape = self.gray.shape[:2]
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
        self.images = self.get_images()

    def get_images(self):
        globbed = glob.glob('*.jpg', root_dir=self.camera_path)
        if not len(globbed):
            return []
        images = []
        for image in globbed:
            image_path = os.path.join(self.camera_path, image)
            images.append(ArucoImage(image_path))
        return images
    
    def charuco_points(self, aruco:ArucoHelper):
        self.object_points = []
        self.charuco_corners = []
        self.charuco_ids = []
        self.shape = None

        for image in self.images:
            if not image.process(aruco):
                print(f'{image.image_path} Error: {image.get_error()}')
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

    def save(self):
        self.save_json()
        self.save_npz()

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

    def save(self):
        for camera in self.get_valid_cameras():
            camera.save()

    def has_calibrations(self):
        return self.get_valid_camera_count() > 0
    
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

    def process(self, aruco:ArucoHelper):
        if not self.has_cameras():
            return False
        if not self.has_calibrations():
            self.calibrations(aruco)
        if not self.has_calibrations():
            return False
        return True
        
class Intrinsics:
    def __init__(self, camera:ArucoCamera):
        self.camera = camera

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Give path to rig or images")
    
    aruco = ArucoHelper()
    rig = ArucoRig(sys.argv[1])

    if not rig.process(aruco):
        print(f'Rig Error: {rig.get_error()}')

    rig.save()

    print('Done')