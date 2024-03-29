import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np


def cut_bounding_box(point_cloud, annotation, annotation_move=[0, 0, 0]):
    """
    Function, which cuts bounding box from point-cloud.
    :param point_cloud: numpy 2D array, original point-cloud
    :param annotation: dictionary, annotation of bounding box which should be cut out
    :param annotation_move: numpy 1D array, row translation vector between annotation and LiDAR
    :return: numpy 2D array, annotations point-cloud
    """
    xc = annotation['center']['x'] - annotation_move[0]
    yc = annotation['center']['y'] - annotation_move[1]
    zc = annotation['center']['z'] - annotation_move[2]
    Q1 = annotation['rotation']['x']
    Q2 = annotation['rotation']['y']
    Q3 = annotation['rotation']['z']
    Q4 = annotation['rotation']['w']
    length = annotation['length']
    width = annotation['width']
    height = annotation['height']

    r = R.from_quat([Q1, Q2, Q3, Q4])
    
    rot_matrix = r.as_dcm()

    bbox = point_cloud[
        rot_matrix[0][0] * point_cloud[:, 0] + rot_matrix[1][0] * point_cloud[:, 1] + rot_matrix[2][
            0] * point_cloud[:,
                 2] <
        rot_matrix[0][0] * (xc + rot_matrix[0][0] * length / 2) + rot_matrix[1][0] * (
                yc + rot_matrix[1][0] * length / 2) + rot_matrix[2][0] *
        (zc + rot_matrix[2][0] * length / 2)]

    bbox = bbox[
        rot_matrix[0][0] * bbox[:, 0] + rot_matrix[1][0] * bbox[:, 1] + rot_matrix[2][0] * bbox[:, 2] >
        rot_matrix[0][0] * (xc - rot_matrix[0][0] * length / 2) + rot_matrix[1][0] * (
                yc - rot_matrix[1][0] * length / 2) + rot_matrix[2][0] *
        (zc - rot_matrix[2][0] * length / 2)]

    bbox = bbox[
        rot_matrix[0][1] * bbox[:, 0] + rot_matrix[1][1] * bbox[:, 1] + rot_matrix[2][1] * bbox[:, 2] <
        rot_matrix[0][1] * (xc + rot_matrix[0][1] * width / 2) + rot_matrix[1][1] * (
                yc + rot_matrix[1][1] * width / 2) + rot_matrix[2][1] *
        (zc + rot_matrix[2][1] * width / 2)]

    bbox = bbox[
        rot_matrix[0][1] * bbox[:, 0] + rot_matrix[1][1] * bbox[:, 1] + rot_matrix[2][1] * bbox[:, 2] >
        rot_matrix[0][1] * (xc - rot_matrix[0][1] * width / 2) + rot_matrix[1][1] * (
                yc - rot_matrix[1][1] * width / 2) + rot_matrix[2][1] *
        (zc - rot_matrix[2][1] * width / 2)]

    bbox = bbox[
        rot_matrix[0][2] * bbox[:, 0] + rot_matrix[1][2] * bbox[:, 1] + rot_matrix[2][2] * bbox[:, 2] <
        rot_matrix[0][2] * (xc + rot_matrix[0][2] * height) + rot_matrix[1][2] * (
                yc + rot_matrix[1][2] * height) + rot_matrix[2][2] *
        (zc + rot_matrix[2][2] * height)]

    bbox = bbox[
        rot_matrix[0][2] * bbox[:, 0] + rot_matrix[1][2] * bbox[:, 1] + rot_matrix[2][2] * bbox[:, 2] >
        rot_matrix[0][2] * (xc - rot_matrix[0][2] * 0) + rot_matrix[1][2] * (
                yc - rot_matrix[1][2] * 0) + rot_matrix[2][2] *
        (zc - rot_matrix[2][2] * 0)]

    return bbox


def separate_bbox(point_cloud, annotation, annotation_move = [0, 0, 0]):

    xc = annotation['center']['x'] - annotation_move[0]
    yc = annotation['center']['y'] - annotation_move[1]
    zc = annotation['center']['z'] - annotation_move[2]
    Q1 = annotation['rotation']['x']
    Q2 = annotation['rotation']['y']
    Q3 = annotation['rotation']['z']
    Q4 = annotation['rotation']['w']
    length = annotation['length']
    width = annotation['width']
    height = annotation['height']

    r = R.from_quat([Q1, Q2, Q3, Q4])
    rot_matrix = r.as_dcm()

    mask_1 = (rot_matrix[0][0] * point_cloud[:, 0] + rot_matrix[1][0] * point_cloud[:, 1] + rot_matrix[2][
            0] * point_cloud[:,
                 2] >
        rot_matrix[0][0] * (xc + rot_matrix[0][0] * length / 2) + rot_matrix[1][0] * (
                yc + rot_matrix[1][0] * length / 2) + rot_matrix[2][0] *
        (zc + rot_matrix[2][0] * length / 2))
    mask_2 = (rot_matrix[0][0] * point_cloud[:, 0] + rot_matrix[1][0] * point_cloud[:, 1] + rot_matrix[2][0] * point_cloud[:, 2] <
              rot_matrix[0][0] * (xc - rot_matrix[0][0] * length / 2) + rot_matrix[1][0] * (
                yc - rot_matrix[1][0] * length / 2) + rot_matrix[2][0] *
              (zc - rot_matrix[2][0] * length / 2))
    mask_3 = (rot_matrix[0][1] * point_cloud[:, 0] + rot_matrix[1][1] * point_cloud[:, 1] + rot_matrix[2][1] * point_cloud[:, 2] >
              rot_matrix[0][1] * (xc + rot_matrix[0][1] * width / 2) + rot_matrix[1][1] * (
                yc + rot_matrix[1][1] * width / 2) + rot_matrix[2][1] *
              (zc + rot_matrix[2][1] * width / 2))
    mask_4 = (rot_matrix[0][1] * point_cloud[:, 0] + rot_matrix[1][1] * point_cloud[:, 1] + rot_matrix[2][1] * point_cloud[:, 2] <
              rot_matrix[0][1] * (xc - rot_matrix[0][1] * width / 2) + rot_matrix[1][1] * (
                yc - rot_matrix[1][1] * width / 2) + rot_matrix[2][1] *
              (zc - rot_matrix[2][1] * width / 2))
    mask_5 = (rot_matrix[0][2] * point_cloud[:, 0] + rot_matrix[1][2] * point_cloud[:, 1] + rot_matrix[2][2] * point_cloud[:, 2] >
              rot_matrix[0][2] * (xc + rot_matrix[0][2] * height) + rot_matrix[1][2] * (
                yc + rot_matrix[1][2] * height) + rot_matrix[2][2] *
              (zc + rot_matrix[2][2] * height))
    mask_6 = (rot_matrix[0][2] * point_cloud[:, 0] + rot_matrix[1][2] * point_cloud[:, 1] + rot_matrix[2][2] * point_cloud[:, 2] <
              rot_matrix[0][2] * (xc - rot_matrix[0][2] * 0) + rot_matrix[1][2] * (
                yc - rot_matrix[1][2] * 0) + rot_matrix[2][2] *
              (zc - rot_matrix[2][2] * 0))

    final_mask = np.ma.mask_or(mask_1, mask_2)
    final_mask = np.ma.mask_or(final_mask, mask_3)
    final_mask = np.ma.mask_or(final_mask, mask_4)
    final_mask = np.ma.mask_or(final_mask, mask_5)
    final_mask = np.ma.mask_or(final_mask, mask_6)

    scene = point_cloud[final_mask]
    bbox = point_cloud[final_mask == False]

    return scene, bbox
