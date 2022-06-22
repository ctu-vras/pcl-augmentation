import numpy as np
import copy
import math
from PIL import Image
from scipy.spatial.transform import Rotation as R

from tools.cut_bbox import cut_bounding_box

RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
DEFAULT = '\033[0m'


def make_dictionary(annotation_array):
    """
    Function, which transforms annotation in array to dictionary
    :param annotation_array: 2D list [[center_x, center_y, center_z],[rotation_x, rotation_y, rotation_z, rotation_w],
    [length, width, height], [class]]
    :return: dictionary, annotation
    """
    center = {'x': annotation_array[0][0], 'y': annotation_array[0][1], 'z': annotation_array[0][2]}
    rotation = {'x': annotation_array[1][0], 'y': annotation_array[1][1], 'z': annotation_array[1][2],
                'w': annotation_array[1][3]}
    annotation_dictionary = {'center': center, 'rotation': rotation, 'length': annotation_array[2][0],
                             'width': annotation_array[2][1], 'height': annotation_array[2][2], 'class': annotation_array[3]}
    return annotation_dictionary


def dictionary2array(annotation_dictionary):
    """
    Function, which transforms annotation in dictionary to array.
    :param annotation_dictionary: dictionary, annotation
    :return: 2D list, annotation
    """
    annotation_array = [[annotation_dictionary['center']['x'], annotation_dictionary['center']['y'], annotation_dictionary['center']['z']],
                        [annotation_dictionary['rotation']['x'], annotation_dictionary['rotation']['y'], annotation_dictionary['rotation']['z'], annotation_dictionary['rotation']['w']],
                        [annotation_dictionary['length'], annotation_dictionary['width'], annotation_dictionary['height']], annotation_dictionary['class']]
    return annotation_array


def rotate_bounding_box_2(bbox_pcl, annotation, rotation=1):
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
        
    z_rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                    [np.sin(rotation), np.cos(rotation), 0],
                    [0, 0, 1]])
    
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


def check_bounding_box(scene_pcl, scene_anno, sample_pcl, sample_anno, ok_surface):
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
    volume = cut_bounding_box(scene_pcl, sample_anno)

    for s in ok_surface:
        volume = volume[volume[:, 7] != s]

    if len(volume) > 0:
        ok = False

    if ok:
        for i in range(len(scene_anno)):
            volume = cut_bounding_box(sample_pcl, scene_anno[i])
            if len(volume) > 0:
                ok = False
                break
    return ok


def correct_height(scene_pcl, sample_pcl, sample_anno, ok_surface):
    """
    Function, which corrects height of additional object.
    :param scene_pcl: numpy 2D array, original point-cloud
    :param sample_pcl: numpy 2D array, additional object point-cloud
    :param sample_anno: dictionary, annotation of additional object
    :return: numpy 2D array, corrected object point-cloud
             dictionary, annotation of bounding box
    """
    sample_anno = dictionary2array(sample_anno)
    surface = np.array([])
    radius = 0.1
    ok = True

    while len(surface) == 0:
        surface = np.array([])
        near_points = scene_pcl[(scene_pcl[:, 0]-sample_anno[0][0])**2+(scene_pcl[:, 1]-sample_anno[0][1])**2 <= radius**2]

        for s in ok_surface:

            if len(near_points[near_points[:, 4] == s]) > 0:
                if len(surface) > 0:
                    surface = np.append(surface, near_points[near_points[:, 4] == s], axis=0)
                else:
                    surface = near_points[near_points[:, 4] == s]

        if len(surface) > 0:
            surface = surface[surface[:, 2] > -3]

        radius += 0.1

        if radius > 5:
            ok = False
            break

    if ok:

        road_level = np.mean(surface, axis=0)[2]
        new_bbox_z_level = road_level
        z_move = new_bbox_z_level - sample_anno[0][2]
        sample_pcl[:, 2] += z_move
        sample_anno[0][2] = new_bbox_z_level

    sample_anno = make_dictionary(sample_anno)

    return sample_pcl, sample_anno, ok


def read_label_line(line):
    """
    Function, which processes line of label txt in KITTI format and transform values to LiDAR coordinates system
    :param line: string, line of label txt in default KITTI format
    :return: dictionary, annotation of object
    """
    annotation_items = line.split(' ')

    sample_class = annotation_items[0]

    sample_x = float(annotation_items[1])
    sample_y = float(annotation_items[2])
    sample_z = float(annotation_items[3])

    sample_height = float(annotation_items[4])
    sample_width = float(annotation_items[6])
    sample_lenght = float(annotation_items[5])

    sample_rotation_z = float(annotation_items[7])

    rot_matrix = [[math.cos(sample_rotation_z), -1 * math.sin(sample_rotation_z), 0],
                  [math.sin(sample_rotation_z), math.cos(sample_rotation_z), 0],
                  [0, 0, 1]]

    r = R.from_dcm(rot_matrix)

    quaternions = r.as_quat()

    save_annotation = [[sample_x, sample_y, sample_z],
                       [quaternions[0], quaternions[1], quaternions[2], quaternions[3]],
                       [sample_width, sample_lenght, sample_height], [sample_class]]

    annotation_dictionary = make_dictionary(save_annotation)

    return annotation_dictionary


def find_possible_places(point_cloud, scene_annotation, sample_data, map, map_move, original_pcl, transformation_matrix, config):
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
    sample_annotation = sample_data['anno']
    sample_annotation = sample_annotation.item()

    sample_annotation = read_label_line(sample_annotation)

    ok_map_surface = config['insertion']['placement'][int(sample_annotation['class'][0])]

    ok_surface = []

    for map_surface in ok_map_surface:
        ok_surface = ok_surface + config['insertion']['placement_labels'][map_surface]

    not_on_road = 0
    object_collision = 0
    ## SEARCH FOR POSIBLE PLACES BY ROTATION
    for rot in range(1, 361):
        on_road = True
        sample_pcl, sample_annotation = rotate_bounding_box_2(sample_pcl, sample_annotation)

        global_position = np.hstack((sample_pcl[:, :3], np.ones((len(sample_pcl), 1)))).T
        global_position = transformation_matrix @ global_position
        global_position = global_position - map_move
        global_position = global_position.astype(np.int)

        global_position = global_position[:, global_position[0, :] < len(map)]
        global_position = global_position[:, global_position[0, :] > -1]
        global_position = global_position[:, global_position[1, :] < len(map[0])]
        global_position = global_position[:, global_position[1, :] > -1]

        for i in range(len(global_position[0])):
            if not (map[global_position[0][i]][global_position[1][i]] in ok_map_surface):
                on_road = False
                break

        if on_road:
            sample_pcl, sample_annotation, near_road = correct_height(original_pcl, sample_pcl, sample_annotation, ok_surface)
            if not near_road:
                #print('Road too far')
                continue

            ok = check_bounding_box(point_cloud, scene_annotation, sample_pcl, sample_annotation, ok_surface)
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
