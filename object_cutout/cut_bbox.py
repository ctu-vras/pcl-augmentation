import numpy as np
import os
import json
from scipy.spatial.transform import Rotation as R


def cut_bounding_box(point_cloud, annotation, annotation_move=[0,0,0]):
    """
    :param point_cloud: original point-cloud
    :param annotation: annotation of bounding box which should be cut out (dictionary)
    :return: annotations point-cloud
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