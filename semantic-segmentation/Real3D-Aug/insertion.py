import json
import os
import numpy as np
import copy
import sys
import glob
import math
import random
import yaml
import time

from tools.closing import *
from tools.find_spot import *
from tools.cut_bbox import separate_bbox
from tools.datasets import *

DEBUG = False

if DEBUG:
    from tools.visualization import *

NUMROW = 112
NUMCOLUMN = 360*4

MAX_NUM_TRIES = 100
SAMPLE_TIMEOUT = False

RGB_CLASS = np.array([[255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 0, 255], [123, 123, 123], [0, 255, 0], [0, 0, 0]])


def create_image(labels, filename):
    """
    Function, which create Image of classes.
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
    # img.show()
    return


def add_space_for_spherical(point_cloud):
    '''
    :param point_cloud: point-cloud as 2D numpy array (N x 5). N is number of points.
    Columns are ordered x, y, z, intensity, semantic label.
    :return: point-cloud as 2D numpy array with space for spherical coordination and space for points FOV index and sample index
    '''
    points_num = len(point_cloud)
    out = np.ones((points_num, 9))*-1
    out[:, 0:3] = point_cloud[:, 0:3]    # x, y, z
    out[:, 6:8] = point_cloud[:, 3:5]    # intensity, label
    return out  # x, y, z, r, azimuth, elevation, intensity, label, FOV-index


def fill_spherical(point_cloud):
    '''
    :param point_cloud: point-cloud as 2D numpy array (N x 10). N is number of points.
    Columns are ordered x, y, z, _, _, _, intensity, semantic label, _, _.
    :return: point-cloud as 2D array with computed spherical coordination and maximum and minimum elevation angle in point-cloud
    Columns are ordered x, y, z, radius, azimuth angle, elevation angle, intensity, semantic label, _, _.
    '''
    point_cloud[:, 3] = np.sqrt(point_cloud[:, 0] ** 2 + point_cloud[:, 1] ** 2 + point_cloud[:, 2] ** 2)  # r (0;infty)
    point_cloud[:, 4] = np.arctan2(point_cloud[:, 1], point_cloud[:, 0]) + np.pi  # azimuth (0;2pi)
    point_cloud[:, 5] = np.arccos(point_cloud[:, 2] / point_cloud[:, 3])  # elevation

    min_elevation_angle = np.min(point_cloud[:, 5])
    max_elevation_angle = np.max(point_cloud[:, 5])

    return point_cloud, max_elevation_angle, min_elevation_angle


def geometrical_front_view(point_cloud, num_row, num_column, max_elevation_angle, min_elevation_angle, sample = False):
    '''
    :param point_cloud: point-cloud as 2D numpy array (N x 10). N is number of points.
    Columns are ordered x, y, z, radius, azimuth angle, elevation angle, intensity, semantic label, _, _.
    :param num_row: front view row resolution
    :param num_column: rront view column resolution
    :param max_elevation_angle: maximal elevation angle in original point-cloud
    :param min_elevation_angle: minimal elevation angle in original point-cloud
    :return: front view, labels for front view and point-cloud as 2D numpy array (N x 10). N is number of points.
    Columns are ordered x, y, z, radius, azimuth angle, elevation angle, intensity, semantic label, FOV point index, _.
    '''
    point_num = len(point_cloud)
    d_elevation = (max_elevation_angle - min_elevation_angle) / num_row       # resolution in row
    d_azimuth = 2 * math.pi / num_column           # resolution in column
    label = np.ones((num_row, num_column)) * -1
    train = np.ones((num_row, num_column)) * 500  # distance

    for i in range(point_num):
        point = point_cloud[i]

        point_row = int((point[5] - min_elevation_angle - 0.00001) / d_elevation)
        point_column = int((point[4]%(2 * math.pi)) / d_azimuth)

        if sample and not(num_row > point_row >= 0):
            continue

        assert num_row > point_row >= 0, f'Rows in FoV went something wrong. Min elevation:{min_elevation_angle}, Max elevation:{max_elevation_angle}, Point elevation:{point[5]}'

        assert num_column > point_column >= 0, f'Column in FoV went something wrong. Point azimuth:{point[4]}'

        if label[point_row][point_column] != -1:

            point_cloud[i][8] = point_row * NUMCOLUMN + point_column

            if train[point_row][point_column] > point[3]:

                train[point_row][point_column] = point[3]  # distance

        else:
            label[point_row][point_column] = 1

            train[point_row][point_column] = point[3]     # distance

            point_cloud[i][8] = point_row * NUMCOLUMN + point_column

    return train, label, point_cloud


def extract_anno(anno_path):
    """
    :param anno_path: address of annotation txt file
    :return: list of annotation directories
    """
    output_list = np.array([])
    txt_file = open(anno_path, 'r')

    end_of_file = False
    while not end_of_file:
        annotation = txt_file.readline()

        if len(annotation) == 0:
            end_of_file = True
            continue

        annotation_dictionary = read_label_line(annotation)

        if len(output_list) == 0:
            output_list = np.array([annotation_dictionary])
        else:
            output_list = np.append(output_list, [annotation_dictionary])

    txt_file.close()
    return output_list


def check_covers(annotation, covered_scene):
    backup_annotation = copy.deepcopy(annotation)
    for a in range(len(annotation)):
        covered_part_of_sample = covered_scene[covered_scene[:, 9] == a]
        if len(covered_part_of_sample) <= annotation[a]['removable_points']:
            annotation[a]['removable_points'] = annotation[a]['removable_points'] - len(covered_part_of_sample)
        else:
            return backup_annotation, False
    return annotation, True


def generate_seed(config):

    if config['insertion']['random']:
        seed = np.zeros(len(config['insertion']['classes']))
        objects = np.random.randint(len(config['insertion']['classes']), size=config['insertion']['number_of_object'])
        for i in objects:
            seed[i] += 1

    else:
        seed = np.array(config['insertion']['number_of_classes'])

    for i in range(len(seed)):
        if seed[i] > 0:
            inserted_class = config['insertion']['classes'][i]
            break

    return seed, inserted_class


def print_status(remaining_objects, config, current_object, frame_name, f_n):
    print(f'Status: {frame_name} for {f_n}')
    classes = config['insertion']['classes']
    names = config['labels']
    for i in range(len(remaining_objects)):
        if remaining_objects[i] <= 0:
            print(f'{GREEN} 0 {names[classes[i]]} remaining.{DEFAULT}')
        elif i == classes.index(current_object):
            print(f'{YELLOW} {remaining_objects[i]} {names[classes[i]]} remaining.{DEFAULT}')
        else:
            print(f'{RED} {remaining_objects[i]} {names[classes[i]]} remaining.{DEFAULT}')


def addjust_map_2(map_data, point_cloud, transformation_matrix):
    # driveable area map
    map = map_data['map']
    map_move = map_data['move']

    point_cloud = point_cloud[point_cloud[:, 2] < 1.5]

    for i in ROAD_INDEXES:
        point_cloud = point_cloud[point_cloud[:, 7] != i]

    hom_points = np.hstack(((point_cloud[:, :3]), np.ones((len(point_cloud), 1)))).T

    hom_points = transformation_matrix @ hom_points

    hom_points = hom_points - map_move

    hom_points = hom_points.astype(np.int)

    for i in range(len(hom_points[0])):
        if map[hom_points[0][i]][hom_points[1][i]] != 0:
            map[hom_points[0][i]][hom_points[1][i]] = 4

    return map, map_move


def dataset_selection():
    quit = False
    dataset = 'SemanticKITTI'
    while not quit:
        print('Choose dataset:')
        print('1 - SemanticKITTI')
        print('2 - Waymo')
        tmp = input()
        if tmp == '1' or tmp == '2':
            quit = True
            if tmp == '1':
                dataset = 'SemanticKITTI'
            else:
                dataset = 'Waymo'
        else:
            print('Wrong input try again')

    sequence = None
    if dataset == 'SemanticKITTI':
        print('SemanticKITTI was chosen')
        if MAC:
            with open('../config/semantic-kitti-mac.yaml', 'r') as file:
                config = yaml.safe_load(file)

                quit = False
                while not quit:
                    print('Choose sequence')
                    sequence = input()
                    if int(sequence) in config['split']['train']:
                        quit = True
                        sequence = f'{int(sequence):02d}'

                dataset_functions = SemanticKITTI(config, sequence)

        else:
            with open('../config/semantic-kitti.yaml', 'r') as file:
                config = yaml.safe_load(file)

                quit = False
                while not quit:
                    print('Choose sequence')
                    sequence = input()
                    if int(sequence) in config['split']['train']:
                        quit = True
                        sequence = f'{int(sequence):02d}'

                dataset_functions = SemanticKITTI(config, sequence)

    elif dataset == 'Waymo':
        print('Waymo was chosen')
        if MAC:
            with open('../config/waymo-mac.yaml', 'r') as file:
                config = yaml.safe_load(file)
                dataset_functions = Waymo(config)

        else:
            with open('../config/waymo.yaml', 'r') as file:
                config = yaml.safe_load(file)
                dataset_functions = Waymo(config)

    return config, dataset_functions, sequence


if __name__ == '__main__':

    config, dataset_functions, sequence = dataset_selection()

    if config['insertion']['random']:
        save_folder = 'random'
    else:
        save_folder = 'chosen'

    save_output_folder = config['path']['output_path']
    data_path = config['path']['dataset_path']
    anno_path = config['path']['annotation_path']
    sample_path = config['path']['bbox_path']

    save_folder, f_n = dataset_functions.create_directories(save_folder)

    sequence = dataset_functions.sequence

    save_folder = f'{save_folder}/{sequence}'

    add_objects_txt_open = False

    maps_path = config['path']['maps_path']
    maps_data = np.load(f'{maps_path}/{sequence}.npz', allow_pickle=True)

    pcl_idx = 0

    while len(dataset_functions) > 0:

        if sequence != dataset_functions.sequence:      # if it is new sequence load appropriate rich map
            sequence = dataset_functions.sequence
            save_folder = save_folder.split('/')
            save_folder[-1] = sequence
            save_folder = '/'.join(save_folder)
            maps_path = config['path']['maps_path']
            maps_data = np.load(f'{maps_path}/{sequence}.npz', allow_pickle=True)

        ## Just for print

        file = dataset_functions.velodyne_list[pcl_idx]

        remaining_objects, inserted_class = generate_seed(config)

        frame_name = file.split('/')[-1].split('.')[0]

        print(f'{sequence}/{frame_name}')

        ##

        if os.path.exists(f'{save_output_folder}/{save_folder}/added_objects/{frame_name}.txt'): # check if scene is done or in progress
            print(f'Already in progress')
            dataset_functions.delete_item(pcl_idx)
            continue

        if add_objects_txt_open:
            add_objects_name.close()
            add_objects_txt_open = False

        assert not add_objects_txt_open, f'Previous txt file for name of added objects was not closed'

        add_objects_name = open(f'{save_output_folder}/{save_folder}/added_objects/{frame_name}.txt', 'w') # txt file of added objects
        add_objects_txt_open = True

        #### SCENE POINT CLOUD
        # original point cloud
        scene_pcl, transform_matrix, anno_location, _, _ = dataset_functions[pcl_idx]   # loading scene data

        original_pcl = copy.deepcopy(scene_pcl)

        scene_annotation = extract_anno(f'{anno_location}')  # creating list of annotation dictionaries from txt file

        ### SCENE FIELD OF VIEW (smooth)
        scene_pcl = add_space_for_spherical(scene_pcl)

        all_visible_parts = []
        some_object_inserted = False
        SAMPLE_TIMEOUT = False
        saved = False

        unplaceble_samples = []

        while np.max(remaining_objects) > 0:    # while some object still need to be added

            scene_pcl, max_elevation, min_elevation = fill_spherical(scene_pcl)     # compute spherical coordination

            scene_train, scene_label, scene_pcl = geometrical_front_view(scene_pcl, NUMROW, NUMCOLUMN, max_elevation, min_elevation)

            scene_train, scene_label = smooth_out(scene_train, scene_label)         # apply closing to FoV

            map, map_move = addjust_map_2(maps_data, scene_pcl, transform_matrix)   # mark occupied places in map (by any scene objects)

            scene_pcl_backup = copy.deepcopy(scene_pcl)

            ### SAMPLE POINT CLOUD
            for i in range(len(remaining_objects)):     # looking for class of next placed object
                if remaining_objects[i] > 0:
                    if inserted_class != config['insertion']['classes'][i]:
                        SAMPLE_TIMEOUT = False
                    inserted_class = config['insertion']['classes'][i]
                    sample_folder = config['labels'][inserted_class]
                    break

            sample_directory_list = glob.glob(f'{sample_path}/{sample_folder}/*.npz')

            match_find = False

            while not match_find:

                if not SAMPLE_TIMEOUT:  # shuffle samples in order to add random one
                    random.shuffle(sample_directory_list)
                    start_index = 0
                    end_index = MAX_NUM_TRIES
                else:   # in order to skip already tried samples (which was unplaceable)
                    start_index += MAX_NUM_TRIES
                    end_index += MAX_NUM_TRIES
                    if end_index > len(sample_directory_list):
                        end_index = len(sample_directory_list)

                for s_index in range(start_index, end_index):
                    sample_file = sample_directory_list[s_index]

                    if match_find:  # previous sample was added
                        break

                    if sample_file.endswith('.npz'):

                        # just for print
                        object_name = sample_file.split('/')[-1].split('.')[0]
                        print(f'\r{object_name}. Try number: {s_index}')
                        #

                        if object_name in unplaceble_samples:
                            print(f'Object was previously unplaceble.')
                            if s_index == end_index-1:
                                remaining_objects[config['insertion']['classes'].index(inserted_class)] -= 1
                                match_find = True
                                break
                            continue

                        unplaceble = True

                        sample_data = np.load(sample_file, allow_pickle=True) # loading sample data

                        possible_sample_pcl, possible_sample_annotation, possible_rotation = find_possible_places(scene_pcl,
                                                                                    scene_annotation, sample_data, map, map_move, original_pcl, transform_matrix, config)

                        if DEBUG:
                            tmp = copy.deepcopy(scene_pcl)
                            tmp[:, 7] = 0
                            
                            for tmp1 in possible_sample_pcl:
                                tmp1[:, 4] = 1
                                tmp = np.append(tmp, add_space_for_spherical(tmp1), axis=0)
                            visualization(tmp)

                        if len(possible_sample_pcl) == 0 and unplaceble:
                            unplaceble_samples.append(object_name)
                            print(f'\rObject: {object_name} was added to unplaceble list')

                        for sample_index in range(len(possible_sample_pcl)):    # computing occlustions for all possible placements
                            sample_pcl = possible_sample_pcl[sample_index]
                            sample_annotation = possible_sample_annotation[sample_index]
                            sample_rotation = possible_rotation[sample_index]
                            scene_pcl = copy.deepcopy(scene_pcl_backup)

                            sample_pcl = add_space_for_spherical(sample_pcl)

                            sample_pcl, _, _ = fill_spherical(sample_pcl)

                            sample_train, sample_label, sample_pcl = geometrical_front_view(sample_pcl, NUMROW, NUMCOLUMN, max_elevation, min_elevation, sample=True)

                            sample_train, sample_label = smooth_out(sample_train, sample_label)

                            first_part = True
                            visible_sample = np.array([])
                            covered_scene = np.array([])

                            indexes = np.where(sample_train < scene_train)  # pixels where sample wold be visible

                            for ind in range(len(indexes[0])):
                                covered_part = scene_pcl[
                                    scene_pcl[:, 8] == indexes[0][ind] * NUMCOLUMN + indexes[1][ind]]  # points in coved pixel
                                scene_pcl = scene_pcl[
                                    scene_pcl[:, 8] != indexes[0][ind] * NUMCOLUMN + indexes[1][ind]]  # all points without covered pixel
                                visible_part = sample_pcl[
                                    sample_pcl[:, 8] == indexes[0][ind] * NUMCOLUMN + indexes[1][ind]]  # sample points in pixel
                                if first_part:
                                    first_part = False
                                    visible_sample = visible_part
                                    covered_scene = covered_part
                                else:
                                    visible_sample = np.append(visible_sample, visible_part, axis=0)
                                    covered_scene = np.append(covered_scene, covered_part, axis=0)

                            if DEBUG:
                                tmp = copy.deepcopy(scene_pcl)
                                tmp[:, 7] = 0
                                tmp = np.append(tmp, visible_sample, axis=0)
                                visualization(tmp)
                                tmp = copy.deepcopy(scene_pcl)
                                tmp[:, 7] = 1
                                _, tmp_label, _ = geometrical_front_view(tmp, NUMROW, NUMCOLUMN, max_elevation,
                                                                         min_elevation)
                                for r in range(NUMROW):
                                    for c in range(NUMCOLUMN):
                                        if tmp_label[r][c] == -1:
                                            tmp_label[r][c] = 0
                                create_image(tmp_label, 'scene_FoV.png')

                                tmp = copy.deepcopy(visible_sample)
                                tmp[:, 7] = 1
                                _, tmp_label, _ = geometrical_front_view(tmp, NUMROW, NUMCOLUMN, max_elevation,
                                                                         min_elevation)
                                for r in range(NUMROW):
                                    for c in range(NUMCOLUMN):
                                        if tmp_label[r][c] == -1:
                                            tmp_label[r][c] = 0
                                create_image(tmp_label, 'sample_FoV.png')

                            assert int(sample_annotation['class'][0]) == inserted_class, f'Inserted class does not match'

                            if len(visible_sample) == 0:
                                print(f'\r{RED}No visible part of {config["labels"][inserted_class]}{DEFAULT} with rotation: {sample_rotation}', end='')

                            elif len(visible_sample) < config['insertion']['min_points'][inserted_class]:
                                print(f'\rIt is visible just {YELLOW}{len(visible_sample)}{DEFAULT} points from {config["labels"][inserted_class]} with rotation: {sample_rotation}. It could be hard case. We try another one.', end='')

                            else:
                                add_objects_name.write(f'{object_name} with rotation: {sample_rotation}\n')
                                print(f'\rIt is visible {GREEN}{len(visible_sample)}{DEFAULT} points from {config["labels"][inserted_class]}. Moving to another scene')
                                match_find = True
                                SAMPLE_TIMEOUT = False
                                unplaceble = False

                                some_object_inserted = True

                                scene_pcl = np.append(scene_pcl, visible_sample, axis=0)

                                remaining_objects[config['insertion']['classes'].index(inserted_class)] -= 1

                                print_status(remaining_objects, config, inserted_class, f'{sequence}/{frame_name}', f_n)

                                if np.max(remaining_objects) > 0:
                                    scene_annotation = np.append(scene_annotation, [sample_annotation])
                                    if len(all_visible_parts) == 0:
                                        all_visible_parts = copy.deepcopy(visible_sample)
                                    else:
                                        all_visible_parts = np.append(all_visible_parts, visible_sample, axis=0)

                                    break

                                else:
                                    if len(all_visible_parts) == 0:
                                        all_visible_parts = copy.deepcopy(visible_sample)
                                    else:
                                        all_visible_parts = np.append(all_visible_parts, visible_sample, axis=0)

                                    # SAVING DATA

                                    dataset_functions.save_data(scene_pcl, all_visible_parts, save_folder, frame_name, pcl_idx)
                                    add_objects_name.close()
                                    add_objects_txt_open = False
                                    saved = True

                                    break

                            if sample_index == len(possible_sample_pcl)-1 and unplaceble:
                                print(f'\rObject: {object_name} was added to unplaceble list')
                                unplaceble_samples.append(object_name)

                    if (s_index == len(
                                sample_directory_list) - 1 or s_index == 3 * MAX_NUM_TRIES) and not match_find:
                        remaining_objects[config['insertion']['classes'].index(inserted_class)] = 0
                        print(f'{RED}TOTAL {inserted_class} insertion time-out.{DEFAULT}')
                        SAMPLE_TIMEOUT = True

                        print_status(remaining_objects, config, inserted_class, f'{sequence}/{frame_name}', f_n)

                    if s_index == end_index-1 and not match_find:
                        remaining_objects[config['insertion']['classes'].index(inserted_class)] -= 1

                        if remaining_objects[config['insertion']['classes'].index(inserted_class)] <= 0:
                            match_find = True

                        print_status(remaining_objects, config, inserted_class, f'{sequence}/{frame_name}', f_n)

                        if np.max(remaining_objects) <= 0 and some_object_inserted:

                            # SAVING DATA
                            dataset_functions.save_data(scene_pcl, all_visible_parts, save_folder, frame_name, pcl_idx)
                            add_objects_name.close()
                            add_objects_txt_open = False
                            saved = True
                            match_find = True

                        quit = False
                        break

        if not some_object_inserted:
            dataset_functions.delete_item(pcl_idx)
            add_objects_name.close()
            add_objects_txt_open = False
            os.remove(f'{save_output_folder}/{save_folder}/added_objects/{frame_name}.txt')

        if some_object_inserted and not saved:

            dataset_functions.save_data(scene_pcl, all_visible_parts, save_folder, frame_name, pcl_idx)
            add_objects_name.close()
            add_objects_txt_open = False
            saved = True

        assert not (some_object_inserted and not saved), f'Some object was inserted but data was not saved.'
