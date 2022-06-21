import numpy as np
from skimage import io


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


def get_calib(calib_file):
    return Calibration(calib_file)


class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']
        self.R0 = calib['R0']
        self.V2C = calib['Tr_velo2cam']

        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)
        
    def cart_to_hom(self, pts):
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom


    def lidar_to_rect(self, pts_lidar):
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        return pts_rect
        
    def rect_to_lidar(self, pts_rect):
        pts_rect_hom = self.cart_to_hom(pts_rect)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))
        V2C_ext[3, 3] = 1
        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]
        
    def rect_to_img(self, pts_rect):
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]
        return pts_img, pts_rect_depth


def get_lidar(lidar_file):
    return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)


def get_fov_flag(pts_rect, img_shape, calib):
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    return pts_valid_flag


def get_image_shape(img_file):
    return np.array(io.imread(img_file).shape[:2], dtype=np.int32)


def cutout_frame(pcl, img_file, calib_file):
    """
    Function, which cuts point-cloud just to relevant part for neural network
    :param pcl: numpy 2D array, original point-cloud with shape N x 4, where N is number of points
    :param img_file: string, address of image file (corresponding with point-cloud)
    :param calib_file: string, address of calibration file (corresponding with point-cloud)
    :return:numpy 2D array, relevant part of point-cloud as 2D numpy array with shape N x 4, where N is number of points
    """

    img_shape = get_image_shape(img_file)
    calib = get_calib(calib_file)
    
    pts_rect = calib.lidar_to_rect(pcl[:, 0:3])
    fov_flag = get_fov_flag(pts_rect, img_shape, calib)
    pcl = pcl[fov_flag]
    return pcl
