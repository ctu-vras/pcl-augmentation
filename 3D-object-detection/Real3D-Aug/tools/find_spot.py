import numpy as np
import copy
import math
from PIL import Image
from scipy.spatial.transform import Rotation as R

from tools.cut_bbox import cut_bounding_box

HIGHLIGHT = False

RGB_CLASS = np.array([[0, 0, 128], [0, 191, 255], [255, 0, 255], [128, 0, 0], [123, 123, 123], [0, 100, 0], [0, 0, 0],
                      [255, 192, 203]])

MINIMAL_OBJECT_POINTS = 15

ROAD_INDEXES = [40, 44, 48]     # Road, Parking, Sidewalk

RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
DEFAULT = '\033[0m'


def create_image(labels, filename):
    """
    Function, which creates Image of classes.
    :param labels: numpy 2D array with classes
    :param filename: string, name of the image
    """

    columns = len(labels[0])
    lines = len(labels)
    rgb = np.zeros(3 * columns * lines).reshape(lines, columns, 3)

    for i in range(lines):
        for j in range(columns):
            labels_idx = int(labels[i][j])

            rgb[i][j] = RGB_CLASS[labels_idx]

    rgb = np.uint8(rgb)
    img = Image.fromarray(rgb, 'RGB')
    img.save(filename)


def make_dictionary(annotation_array):
    """
    Function, which transforms annotation in array to dictionary
    :param annotation_array: 2D list [[center_x, center_y, center_z],[rotation_x, rotation_y, rotation_z, rotation_w],
    [length, width, height], [class]]
    :return: dictionary, annotation
    """
    center = {'x': annotation_array[0][0], 'y': annotation_array[0][1], 'z': annotation_array[0][2]}
    rotation = {'x': annotation_array[1][0], 'y': annotation_array[1][1], 'z': annotation_array[1][2], 'w': annotation_array[1][3]}
    annotation_dictionary = {'center': center, 'rotation': rotation, 'length': annotation_array[2][0],
                             'width': annotation_array[2][1], 'height': annotation_array[2][2], 'class': annotation_array[3][0], 'removable_points': annotation_array[4][0]}
    return annotation_dictionary


def dictionary2array(annotation_dictionary):
    """
    Function, which transforms annotation in dictionary to array.
    :param annotation_dictionary: dictionary, annotation
    :return: 2D list, annotation
    """
    annotation_array = [[annotation_dictionary['center']['x'], annotation_dictionary['center']['y'], annotation_dictionary['center']['z']],
                        [annotation_dictionary['rotation']['x'], annotation_dictionary['rotation']['y'], annotation_dictionary['rotation']['z'], annotation_dictionary['rotation']['w']],
                        [annotation_dictionary['length'], annotation_dictionary['width'], annotation_dictionary['height']],
                        [annotation_dictionary['class']], [annotation_dictionary['removable_points']]]
    return annotation_array


def rotate_bounding_box(bbox_pcl, annotation, rotation=1):
    """
    Function, which rotate point-cloud and annotation around Z point-cloud axes.
    :param bbox_pcl: numpy 2D array, point cloud inside bounding box
    :param annotation: dictionary, bounding box annotation
    :param rotation: float, angle to rotate bounging box (in degrees) default is 1 degree
    :return: numpy 2D array, rotated point-cloud
             dictionary, annotation of bounding box
    """
    annotation = dictionary2array(annotation)
    rotation = np.deg2rad(rotation)
    r = R.from_quat(annotation[1])
    rot_matrix = r.as_dcm()
    # print(rot_matrix)
    z_rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                    [np.sin(rotation), np.cos(rotation), 0],
                    [0, 0, 1]])
    # print(z_rot_matrix)
    final_rotation = np.dot(rot_matrix, z_rot_matrix)
    ft = R.from_dcm(final_rotation)
    annotation[1] = ft.as_quat()
    position = np.array([[annotation[0][0]], [annotation[0][1]], [annotation[0][2]]])
    position = np.dot(z_rot_matrix, position)
    annotation[0][0] = position[0][0]
    annotation[0][1] = position[1][0]
    annotation[0][2] = position[2][0]

    bbox_pcl[:, :3] = (z_rot_matrix @ bbox_pcl[:, :3].T).T

    annotation = make_dictionary(annotation)

    return bbox_pcl, annotation


def check_bounding_box(scene_pcl, scene_anno, sample_pcl, sample_anno):
    """
    Function, which checks if object, which we want to added would have collision with same object in original point-cloud
    :param scene_pcl: numpy 2D array, original point-cloud
    :param scene_anno: dictionary, annotation of all objects in point-cloud
    :param sample_pcl: numpy 2D array, additional object point-cloud
    :param sample_anno: dictionary, annotation of additional object in form of dictionary
    :return: True if tested position does not cause collision
             False otherwise
    """
    ok = True
    tmp = cut_bounding_box(scene_pcl, sample_anno)
    tmp = tmp[tmp[:, 7] == 1]

    if sample_anno['class'] == 'Pedestrian':
        tmp = tmp[tmp[:, 2] >= sample_anno['center']['z'] + 0.1]

    if len(tmp) > 0:
        ok = False

    if ok:
        for i in range(len(scene_anno)):
            tmp = cut_bounding_box(sample_pcl, scene_anno[i])
            if len(tmp) > 0:
                ok = False
                break
    return ok


def correct_height(scene_pcl, sample_pcl, sample_anno):
    """
    Function, which corrects height of additional object.
    :param scene_pcl: numpy 2D array, original point-cloud
    :param sample_pcl: numpy 2D array, additional object point-cloud
    :param sample_anno: dictionary, annotation of additional object
    :return: numpy 2D array, corrected object point-cloud
             dictionary, annotation of bounding box
    """
    sample_anno = dictionary2array(sample_anno)
    tmp = np.array([])
    radius = 0.1

    while len(tmp) == 0:
        tmp = scene_pcl[(scene_pcl[:, 0]-sample_anno[0][0])**2+(scene_pcl[:, 1]-sample_anno[0][1])**2 <= radius**2]
        tmp = tmp[tmp[:, 4] == 40]
        tmp = tmp[tmp[:, 2] > -3]
        radius += 0.1
    # print('R = ', radius)

    road_level = np.mean(tmp, axis=0)[2]
    new_bbox_z_level = road_level
    z_move = new_bbox_z_level - sample_anno[0][2]
    sample_pcl[:, 2] += z_move
    sample_anno[0][2] = new_bbox_z_level
    if radius > 5:
        ok = False
    else:
        ok = True

    sample_anno = make_dictionary(sample_anno)

    return sample_pcl, sample_anno, ok


def compute_max_remove_points_coefficient(truncated, occluded):
    """
    :param truncated: Float from 0 (non-truncated) to 1 (truncated),
                      where truncated refers to the object leaving image boundaries
    :param occluded:  Integer (0,1,2,3) indicating occlusion state:
                      0 = fully visible, 1 = partly occluded
                      2 = largely occluded, 3 = unknown
    :return:Integer indicating how many points can be removed from object to not become too hard for neural network
    """
    if occluded == 2 or occluded == 3 or truncated > 0.5:
        return 0
    elif occluded == 1:
        if truncated >= 0.25:
            return 0
        else:
            return 0.25 - truncated
    else:
        if truncated >= 0.5:
            return 0
        else:
            return 0.5 - truncated


def read_label_line(line):
    """
    Function, which processes line of label txt in KITTI format and transform values to LiDAR coordinates system
    :param line: string, line of label txt in default KITTI format
    :return: dictionary, annotation of object
    """
    annotation_items = line.split(' ')

    sample_class = annotation_items[0]

    ### SAMPLE OCCLUSION
    if sample_class == 'Car' or sample_class == 'Cyclist' or sample_class == 'Pedestrian':
        max_remove_points_coefficient = compute_max_remove_points_coefficient(float(annotation_items[1]), int(annotation_items[2]))
    else:
        max_remove_points_coefficient = 1

    ### CAMERA COORDINATIONS
    sample_height = float(annotation_items[8])
    sample_width = float(annotation_items[9])
    sample_length = float(annotation_items[10])

    sample_x = float(annotation_items[11])
    sample_y = float(annotation_items[12])
    sample_z = float(annotation_items[13])

    sample_rotation_y = float(annotation_items[14])
    ###

    ### LiDAR COORDINATIONS
    corrected_height = sample_height + 0.1
    corrected_width = sample_length + 0.1
    corrected_length = sample_width + 0.1

    corrected_x = float(sample_z) + 0.27
    corrected_y = float(sample_x) * -1
    corrected_z = float(sample_y) * -1 - 0.08

    z_rot = float(sample_rotation_y) * -1
    ###

    rot_matrix = [[math.cos(z_rot), -1 * math.sin(z_rot), 0],
                  [math.sin(z_rot), math.cos(z_rot), 0],
                  [0, 0, 1]]

    r = R.from_dcm(rot_matrix)

    quaternions = r.as_quat()

    annotation_dictionary = [[corrected_x, corrected_y, corrected_z],
                             [quaternions[0], quaternions[1], quaternions[2], quaternions[3]],
                             [corrected_length, corrected_width, corrected_height],
                             [sample_class], [max_remove_points_coefficient]]

    annotation_dictionary = make_dictionary(annotation_dictionary)

    return annotation_dictionary


def find_possible_places(point_cloud, scene_annotation, sample_data, map_data, original_pcl):
    """
    Function, which finds all possible placement of additional object
    :param point_cloud: numpy 2D array, original point-cloud
    :param scene_annotation: dictionary, annotation of all objcet in point-cloud
    :param sample_data: data about additional object (point-cloud and annotation)
                        ['pcl'] - numpy 2D array, objects point-cloud
                        ['anno'] - string, objects annotation
    :param map_data: ['map'] - numpy 2D array
                     ['min_x'],['min_y'] - float, map move
                        Values in map:
                                    0-background
                                    1-road / sidewalk
    :return: output_pcl - list of numpy 2D array, list of possible object point-clouds
             output_annotation - list of dictionaries, list of possible object annotations
             output_rotation - list, list of possible object rotation
             Note: indexes of all 3 outputs correspond to each other
    """
    output_pcl = []
    output_annotation = []
    output_rotation = []
    ### LOADING DATA
    # sample point cloud
    sample_pcl = sample_data['pcl']
    sample_pcl[:, 4] = 1
    sample_annotation = sample_data['anno']
    sample_annotation = sample_annotation.item()

    sample_annotation = read_label_line(sample_annotation)
    sample_annotation['removable_points'] = sample_annotation['removable_points'] * len(sample_pcl)

    if len(sample_pcl) - sample_annotation['removable_points'] < MINIMAL_OBJECT_POINTS:
        sample_annotation['removable_points'] = len(sample_pcl) - MINIMAL_OBJECT_POINTS

    # driveable area map
    map = map_data['map']
    map_move = np.array([map_data['min_x'], map_data['min_y']])

    not_on_road = 0
    object_collision = 0
    ## SEARCH FOR POSIBLE PLACES BY ROTATION
    for rot in range(1, 361):
        on_road = False
        sample_pcl, sample_annotation = rotate_bounding_box(sample_pcl, sample_annotation)

        for sample_point in sample_pcl:

            global_position = np.array([sample_point[0], sample_point[1]]) - map_move

            if global_position[0] < 0 or global_position[0] >= map.shape[0] or global_position[1] < 0 or global_position[1] >= map.shape[1]:
                continue

            elif map[int(global_position[0])][int(global_position[1])] == 1:
                on_road = True

            elif map[int(global_position[0])][int(global_position[1])] != 1:
                on_road = False
                break

        if on_road:
            sample_pcl, sample_annotation, near_road = correct_height(original_pcl, sample_pcl, sample_annotation)
            if not near_road:
                #print('Road too far')
                continue

            ok = check_bounding_box(point_cloud, scene_annotation, sample_pcl, sample_annotation)
            if ok:
                if len(output_pcl) == 0:
                    output_pcl = [copy.deepcopy(sample_pcl)]
                    output_annotation = [copy.deepcopy(sample_annotation)]
                    output_rotation = [rot]
                else:
                    output_pcl.append(copy.deepcopy(sample_pcl))
                    output_annotation.append(copy.deepcopy(sample_annotation))
                    output_rotation.append(rot)
            else:
                object_collision += 1
        else:
            not_on_road += 1

    print(f'From 360 possibilities, {YELLOW}{not_on_road}{DEFAULT} was not on road, {RED}{object_collision}{DEFAULT} has collision with another object, and {GREEN}{len(output_pcl)}{DEFAULT} was possible.')

    return output_pcl, output_annotation, output_rotation