#import open3d as o3d
import json
import os
import numpy as np
import copy
import sys
import glob
import math
import random

from tools.closing import *
from tools.find_spot import *
from tools.cutout import *
from tools.cut_bbox import separate_bbox

DEBUG = False

NUMROW = 112
NUMCOLUMN = 360*4

MAX_NUM_TRIES = 100
SAMPLE_TIMEOUT = False
TEST = False

RGB_CLASS = np.array([[255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 0, 255], [123, 123, 123], [0, 255, 0], [0, 0, 0]])

DATA_PATH = '/datagrid/personal/vacekpa2/students/kitti/training_default'

SAMPLE_PATH = '/datagrid/personal/vacekpa2/students/sebekpe1/KITTI/test_objects'

TRAIN_TXT_PATH = '/datagrid/personal/vacekpa2/students/kitti/training_default/train.txt'

SAVE_OUTPUT_FOLDER = '/datagrid/personal/vacekpa2/students/kitti/training_default'


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
    out = np.ones((points_num, 10))*-1
    out[:, 0:3] = point_cloud[:, 0:3]    # x, y, z
    out[:, 6:8] = point_cloud[:, 3:5]    # intensity, label
    return out  # x, y, z, r, azimuth, elevation, intensity, label, FOV-index, object-index


def remove_space_for_spherical(point_cloud):
    '''
    :param point_cloud: point-cloud as 2D numpy array with space for spherical coordination
    (format of add_space_for_spherical output)
    :return: point-cloud as 2D numpy array (N x 4). Where N is number of points.
    Columns are ordered x, y, z. intensity
    '''
    points_num = len(point_cloud)
    out = np.ones((points_num, 4))*-1
    out[:, 0:3] = point_cloud[:, 0:3]   # X, Y, Z
    out[:, 3] = point_cloud[:, 6]       # Intensity
    return out


def fill_sample_indexes(point_cloud, annotations):
    for a in range(len(annotations)):
        point_cloud, bbox = separate_bbox(point_cloud, annotations[a])
        bbox[:, 9] = a
        annotations[a]['removable_points'] = annotations[a]['removable_points'] * len(bbox)

        if annotations[a]['class'] == 'Car' or annotations[a]['class'] == 'Cyclist' or annotations[a][
            'class'] == 'Pedestrian':
            if len(bbox) < MINIMAL_OBJECT_POINTS:
                annotations[a]['removable_points'] = 0
            elif len(bbox) - annotations[a]['removable_points'] < MINIMAL_OBJECT_POINTS:
                annotations[a]['removable_points'] = len(bbox) - MINIMAL_OBJECT_POINTS

        point_cloud = np.append(point_cloud, bbox, axis=0)
    return point_cloud, annotations


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
    train = np.ones((num_row, num_column)) * 500   # distance

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
                label[point_row][point_column] = point[7]  # label

                train[point_row][point_column] = point[3]  # distance

        else:
            label[point_row][point_column] = point[7]     # label

            train[point_row][point_column] = point[3]     # distance

            point_cloud[i][8] = point_row * NUMCOLUMN + point_column

    return train, label, point_cloud


def extract_anno(anno_path):
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


def create_annotation_line(original_string, new_annotation_dict, rotation):
    rotation = np.deg2rad(rotation)

    center_x = new_annotation_dict['center']['x']
    center_y = new_annotation_dict['center']['y']
    center_z = new_annotation_dict['center']['z']

    original_string = original_string.item()
    original_items = original_string.split(' ')
    sample_class = new_annotation_dict['class']
    sample_truncated = original_items[1]
    sample_occluded = f'3'
    sample_bbox = f'{original_items[4]} {original_items[5]} {original_items[6]} {original_items[7]}'
    sample_dimensions = f'{original_items[8]} {original_items[9]} {original_items[10]}'
    sample_location = f'{(center_y*-1):.02f} {(center_z*-1)-0.08:.02f} {center_x-0.27:.02f}'

    sample_rotation_y = float(original_items[14]) - rotation

    if sample_rotation_y < -np.pi:
        sample_rotation_y += 2 * np.pi
    elif sample_rotation_y > np.pi:
        sample_rotation_y -= 2 * np.pi

    assert -np.pi <= sample_rotation_y <= np.pi, f'Error in range of sample_rotation_y in file: {file}. Sample_rotation_y = {sample_rotation_y}'

    sample_alpha = (np.arctan2((center_y*-1), center_x-0.27) * -1) + sample_rotation_y

    if sample_alpha < -np.pi:
        sample_alpha += 2 * np.pi
    elif sample_alpha > np.pi:
        sample_alpha -= 2 * np.pi

    assert -np.pi <= sample_alpha <= np.pi, f'Error in range of alpha in file: {file}. Alpha = {sample_alpha}'

    sample_alpha = f'{sample_alpha:.02f}'

    sample_rotation_y = f'{sample_rotation_y:.02f}'

    return f'{sample_class} {sample_truncated} {sample_occluded} {sample_alpha} {sample_bbox} {sample_dimensions} {sample_location} {sample_rotation_y}\n'


def create_annotation(old_address, new_address, additional_annotations_lines):
    old_txt = open(old_address, 'r')
    new_txt = open(new_address, 'w')

    end_of_file = False
    while not end_of_file:
        annotation = old_txt.readline()

        if len(annotation) == 0:
            end_of_file = True
            for a in additional_annotations_lines:
                new_txt.write(a)
            continue

        new_txt.write(annotation)

    old_txt.close()
    new_txt.close()


def color_placement_area(point_cloud, map_data):
    map = map_data['map']
    map_move = np.array([map_data['min_x'], map_data['min_y']])

    for i in range(len(point_cloud)):
        position_x = point_cloud[i][0] - map_move[0]
        position_y = point_cloud[i][1] - map_move[1]
        if map[int(position_x)][int(position_y)] == 1:
            point_cloud[i][7] = 2
        else:
            point_cloud[i][7] = 0

    return point_cloud


def choose_insertion():
    quit = False
    save_folder = None

    inserted_class = ''
    num_cars = 0
    num_bikes = 0
    num_pedestrian = 0

    while not quit:
        print(f'Choose insertion:')
        print(f'     Car ---------> 1')
        print(f'     Bikes -------> 2')
        print(f'     Pedestrian --> 3')
        print(f'     Custom ------> 4')
        choice = input()
        try:
            choice = int(choice)
            if choice == 1:
                print(f'Car insertion')
                num_cars = 3
                save_folder = f'car_insertion_{num_cars:02d}_no_cover_test_data'
                inserted_class = 'Car'
                quit = True

            elif choice == 2:
                print(f'Bikes insertion')
                num_bikes = 7
                save_folder = f'bikes_insertion_{num_bikes:02d}_control_cover'
                inserted_class = 'Cyclist'
                quit = True

            elif choice == 3:
                print(f'Pedestrian insertion')
                num_pedestrian = 1
                save_folder = f'pedestrian_insertion_{num_pedestrian:02d}_no_cover'
                inserted_class = 'Pedestrian'
                quit = True

            elif choice == 4:
                print(f'Custom insertion')
                num_cars = 0
                num_bikes = 5
                num_pedestrian = 5
                save_folder = f'custom_insertion_no_cover_{num_cars}_{num_bikes}_{num_pedestrian}'
                inserted_class = 'Cyclist'
                quit = True

            else:
                print(f'Wrong input try again.')
        except:
            print(f'Wrong input try again.')

    return save_folder, inserted_class, num_cars, num_bikes, num_pedestrian, choice


def create_directories(save_folder):
    folder_number = 0

    if not os.path.exists(f'{SAVE_OUTPUT_FOLDER}/{save_folder}'):
        os.mkdir(f'{SAVE_OUTPUT_FOLDER}/{save_folder}')

    if os.path.exists(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/{folder_number:02d}'):
        print(f'Default save folder is already existing do you want change name? [yes/no]')
        quit = False
        while not quit:
            choice = input()
            if choice == 'yes':
                quit1 = False
                while not quit1:
                    print(f'Write name of the save folder.')
                    folder_number = input()
                    try:
                        folder_number = int(folder_number)

                        if not 0 <= folder_number <= 99:
                            print(f'Input must be number between 0 and 99')
                            continue
                    except:
                        print(f'Input must be number between 0 and 99')
                        continue
                    print(f'Is this what you want: \'{folder_number:02d}\'? [yes/no]')
                    choice = input()

                    if choice == 'yes':
                        quit = True
                        quit1 = True
                        if not os.path.exists(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/{folder_number:02d}'):
                            os.mkdir(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/{folder_number:02d}')

            elif choice == 'no':
                quit = True

            else:
                print(f'Wrong input. Try again.')

    else:
        os.mkdir(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/{folder_number:02d}')

    save_folder = f'{save_folder}/{folder_number:02d}'

    if not os.path.exists(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/data'):
        os.mkdir(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/data')

    if not os.path.exists(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/check'):
        os.mkdir(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/check')

    if not os.path.exists(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/label_2'):
        os.mkdir(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/label_2')

    if not os.path.exists(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/added_objects'):
        os.mkdir(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/added_objects')

    return save_folder


def create_velodyne_list():
    quit = False
    skip_scenes = 0
    order = False

    while not quit:
        print(f'Do you want to reverse order of making scenes?[yes/no] or skip scene?[0;7000] (1/4 = 1842; 1/2 = 3699)')
        tmp = input()
        if tmp == 'no':
            quit = True
            order = False
        elif tmp == 'yes':
            quit = True
            order = True
        else:
            try:
                tmp = int(tmp)
                skip_scenes = tmp
                print(f'Skiping scenes lower then {skip_scenes}')
                quit = True
            except:
                print(f'Wrong input try again.')

    quit = False

    velodyne_address = glob.glob(f'{DATA_PATH}/velodyne/*.bin')

    train_txt = open(TRAIN_TXT_PATH)
    current_directory = []
    while not quit:
        frame_name = train_txt.readline()

        if len(frame_name) == 0:
            train_txt.close()
            quit = True

        else:
            frame_name = int(frame_name)
            if frame_name > skip_scenes or TEST:
                current_directory.append(f'{DATA_PATH}/velodyne/{frame_name:06d}.bin')

    if TEST:
        for f in current_directory:
            velodyne_address.remove(f)

        if skip_scenes > 0:
            for i in range(len(velodyne_address)-1, -1, -1):
                if int(velodyne_address[i].split('/')[-1].split('.')[0]) < skip_scenes:
                    velodyne_address.remove(velodyne_address[i])

        current_directory = velodyne_address

    current_directory.sort(reverse=order)

    return current_directory


def save_data(point_cloud, added_points, added_annotations, folder, name):
    point_cloud = remove_space_for_spherical(point_cloud)
    added_points = remove_space_for_spherical(added_points)

    ## CREATING ANNOTATION

    create_annotation(old_address=f'{DATA_PATH}/label_2/{name}.txt',
                      new_address=f'{SAVE_OUTPUT_FOLDER}/{folder}/label_2/{name}.txt',
                      additional_annotations_lines=added_annotations)

    ## SAVING PCL TO .BIN
    save_file = open(f'{SAVE_OUTPUT_FOLDER}/{folder}/data/{name}.bin', 'wb')
    save_file.write(point_cloud.astype(np.float32))
    save_file.close()

    check_file = open(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/check/{frame_name}.bin', 'wb')
    check_file.write(added_points.astype(np.float32))
    check_file.close()


def check_covers(annotation, covered_scene):
    backup_annotation = copy.deepcopy(annotation)
    for a in range(len(annotation)):
        covered_part_of_sample = covered_scene[covered_scene[:, 9] == a]
        if len(covered_part_of_sample) <= annotation[a]['removable_points']:
            annotation[a]['removable_points'] = annotation[a]['removable_points'] - len(covered_part_of_sample)
        else:
            return backup_annotation, False
    return annotation, True


def create_read_me(save_folder, insertion_idx, inserted_objects):
    txt = open(f'{save_folder}/setting.txt', 'w')
    txt.write(f'Insertion added medium and hard\n')

    if insertion_idx == 1:
        txt.write(f'Insertion {inserted_objects[0]} cars\n')
    elif insertion_idx == 2:
        txt.write(f'Insertion {inserted_objects[1]} bikes\n')
    elif insertion_idx == 3:
        txt.write(f'Insertion {inserted_objects[2]} pedestrians\n')
    elif insertion_idx == 4:
        txt.write(f'Insertion {inserted_objects[0]} cars, {inserted_objects[1]} bikes, {inserted_objects[2]} pedestrians\n')
    else:
        assert False, f'Unrecognized insertion'
    txt.close()


if __name__ == '__main__':

    save_folder, inserted_class, num_cars, num_bikes, num_pedestrian, insertion_idx = choose_insertion()

    save_folder = create_directories(save_folder)

    create_read_me(f'{SAVE_OUTPUT_FOLDER}/{save_folder}', insertion_idx, [num_cars, num_bikes, num_pedestrian])

    velodyne_list = create_velodyne_list()

    quit = False

    add_objects_txt_open = False

    while not quit:

        quit = True

        for file in velodyne_list:
            if file.endswith(f'.bin'):
                print(f'{file}')

                remaining_car = num_cars
                remaining_bikes = num_bikes
                remaining_pedestrians = num_pedestrian

                frame_name = file.split('/')[-1].split('.')[0]
                if os.path.exists(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/added_objects/{frame_name}.txt'):
                    print(f'Already done')
                    continue

                map_data_sidewalk = np.load(f'{DATA_PATH}/maps/pedestrian_area/npz/{frame_name}.npz', allow_pickle=True)
                map_data_road = np.load(f'{DATA_PATH}/maps/road_maps/npz/{frame_name}.npz', allow_pickle=True)

                calib_file = f'{DATA_PATH}/calib/{frame_name}.txt'
                img_file = f'{DATA_PATH}/image_2/{frame_name}.png'

                if add_objects_txt_open:
                    add_objects_name.close()
                    add_objects_txt_open = False

                assert not add_objects_txt_open, f'Previous txt file for name of added objects was not closed'

                add_objects_name = open(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/added_objects/{frame_name}.txt', 'w')
                add_objects_txt_open = True

                #### SCENE POINT CLOUD
                # original point cloud
                scene_pcl = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
                semantic_labels = np.fromfile(f'{DATA_PATH}/semantic_labels/labels/{frame_name}.label', dtype=np.uint32).reshape(-1, 1)
                semantic_labels = semantic_labels & 0xFFFF
                scene_pcl = np.hstack((scene_pcl, semantic_labels))

                original_pcl = copy.deepcopy(scene_pcl)

                scene_annotation = extract_anno(f'{DATA_PATH}/label_2/{frame_name}.txt')

                ###

                for i in range(len(scene_pcl)):
                    if scene_pcl[i][4] != 40 and scene_pcl[i][4] != 44 and scene_pcl[i][4] != 48: # road, parking, sidewalk
                        scene_pcl[i][4] = 1   # background

                ### SCENE FIELD OF VIEW (smooth)
                scene_pcl = add_space_for_spherical(scene_pcl)

                scene_pcl, scene_annotation = fill_sample_indexes(scene_pcl, scene_annotation)

                additional_annotations_lines = []
                all_visible_parts = []
                some_object_inserted = False
                SAMPLE_TIMEOUT = False
                saved = False

                unplaceble_samples = []

                while remaining_car > 0 or remaining_bikes > 0 or remaining_pedestrians > 0:

                    scene_pcl, max_elevation, min_elevation = fill_spherical(scene_pcl)

                    scene_train, scene_label, scene_pcl = geometrical_front_view(scene_pcl, NUMROW, NUMCOLUMN, max_elevation, min_elevation)

                    scene_train, scene_label = smooth_out(scene_train, scene_label)
                    ###
                    if DEBUG:
                        tmp_label = copy.deepcopy(scene_label)
                        for r in range(NUMROW):
                            for c in range(NUMCOLUMN):
                                if tmp_label[r][c] == -1:
                                    tmp_label[r][c] = 0
                                else:
                                    tmp_label[r][c] = 1
                        create_image(tmp_label, 'output/smooth_scene.png')

                    scene_pcl_backup = copy.deepcopy(scene_pcl)

                    ### SAMPLE POINT CLOUD
                    if remaining_car > 0:
                        sample_folder = '/kitti_cars'
                        if inserted_class != 'Car':
                            SAMPLE_TIMEOUT = False
                        inserted_class = 'Car'
                    elif remaining_bikes > 0:
                        sample_folder = '/kitti_bikes'
                        if inserted_class != 'Cyclist':
                            SAMPLE_TIMEOUT = False
                        inserted_class = 'Cyclist'
                    elif remaining_pedestrians > 0:
                        sample_folder = '/kitti_pedestrians'
                        if inserted_class != 'Pedestrian':
                            SAMPLE_TIMEOUT = False
                        inserted_class = 'Pedestrian'

                    sample_directory_list = glob.glob(f'{SAMPLE_PATH}{sample_folder}/*.npz')

                    match_find = False

                    while not match_find:
                        if not SAMPLE_TIMEOUT:
                            random.shuffle(sample_directory_list)
                            start_index = 0
                            end_index = MAX_NUM_TRIES
                        else:
                            start_index += MAX_NUM_TRIES
                            end_index += MAX_NUM_TRIES
                            if end_index > len(sample_directory_list):
                                end_index = len(sample_directory_list)

                        for s_index in range(start_index, end_index):
                            sample_file = sample_directory_list[s_index]

                            if match_find:
                                break

                            if sample_file.endswith('.npz'):

                                ### LOADING SAMPLE POINT CLOUD
                                object_name = sample_file.split('/')[-1].split('.')[0]
                                print(f'{object_name}. Try number: {s_index}')

                                # if object_name in unplaceble_samples:
                                #     print(f'Object was previously unplaceble.')
                                #     if s_index == end_index-1 and end_index == len(sample_directory_list):
                                #         if inserted_class == 'Car':
                                #             remaining_car -= 1
                                #             match_find = True
                                #             break
                                #         elif inserted_class == 'Cyclist':
                                #             remaining_bikes -= 1
                                #             match_find = True
                                #             break
                                #         elif inserted_class == 'Pedestrian':
                                #             remaining_pedestrians -= 1
                                #             match_find = True
                                #             break
                                #     continue

                                unplaceble = True

                                sample_data = np.load(sample_file, allow_pickle=True)

                                if inserted_class == 'Car' or inserted_class == 'Cyclist':
                                    possible_sample_pcl, possible_sample_annotation, possible_rotation = find_possible_places(scene_pcl,
                                                                                            scene_annotation, sample_data, map_data_road, original_pcl)
                                elif inserted_class == 'Pedestrian':
                                    possible_sample_pcl, possible_sample_annotation, possible_rotation = find_possible_places(scene_pcl,
                                                                                            scene_annotation, sample_data, map_data_sidewalk, original_pcl)

                                if DEBUG:
                                    tmp = copy.deepcopy(scene_pcl)
                                    tmp[:, 7] = 0
                                    for tmp1 in possible_sample_pcl:
                                        tmp1[:, 4] = 1
                                        tmp = np.append(tmp, add_space_for_spherical(tmp1), axis=0)
                                    visualization(tmp)

                                if len(possible_sample_pcl) == 0 and unplaceble:
                                    unplaceble_samples.append(object_name)
                                    print(f'Object: {object_name} was added to unplaceble list')

                                for sample_index in range(len(possible_sample_pcl)):
                                    sample_pcl = possible_sample_pcl[sample_index]
                                    sample_annotation = possible_sample_annotation[sample_index]
                                    sample_rotation = possible_rotation[sample_index]
                                    scene_pcl = copy.deepcopy(scene_pcl_backup)

                                    original_len_of_sample = len(sample_pcl)

                                    visible_num_points_in_cutout = len(cutout_frame(sample_pcl, img_file, calib_file))

                                    num_point_removed_by_cutout = 0

                                    if visible_num_points_in_cutout < MINIMAL_OBJECT_POINTS or len(sample_pcl) - visible_num_points_in_cutout > sample_annotation['removable_points'] or (len(sample_pcl) - sample_annotation['removable_points'] * 0.3) < MINIMAL_OBJECT_POINTS:
                                        if sample_index == len(possible_sample_pcl) - 1 and unplaceble:
                                            unplaceble_samples.append(object_name)
                                            print(f'Object: {object_name} was added to unplaceble list')
                                        continue
                                    else:
                                        num_point_removed_by_cutout = len(sample_pcl) - visible_num_points_in_cutout
                                        sample_pcl = cutout_frame(sample_pcl, img_file, calib_file)

                                    sample_pcl = add_space_for_spherical(sample_pcl)

                                    sample_pcl, _, _ = fill_spherical(sample_pcl)

                                    sample_train, sample_label, sample_pcl = geometrical_front_view(sample_pcl, NUMROW, NUMCOLUMN, max_elevation, min_elevation, sample=True)

                                    sample_train, sample_label = smooth_out(sample_train, sample_label)

                                    first_part = True
                                    visible_sample = np.array([])
                                    covered_scene = np.array([])

                                    # for row in range(NUMROW):
                                    #     for column in range(NUMCOLUMN):
                                    #         if sample_label[row][column] == 1:
                                    #             if scene_train[row][column][0] > sample_train[row][column][0] or scene_label[row][column] == -1:
                                    #                 covered_part = scene_pcl[scene_pcl[:, 8] == row * NUMCOLUMN + column]   # points in coved pixel
                                    #                 scene_pcl = scene_pcl[scene_pcl[:, 8] != row * NUMCOLUMN + column]      # all without covered pixel
                                    #                 visible_part = sample_pcl[sample_pcl[:, 8] == row * NUMCOLUMN + column] # sample points in pixel
                                    #                 if first_part:
                                    #                     first_part = False
                                    #                     visible_sample = visible_part
                                    #                     covered_scene = covered_part
                                    #                 else:
                                    #                     visible_sample = np.append(visible_sample, visible_part, axis=0)
                                    #                     covered_scene = np.append(covered_scene, covered_part, axis=0)

                                    indexes = np.where(sample_train < scene_train)

                                    for ind in range(len(indexes[0])):
                                        covered_part = scene_pcl[
                                            scene_pcl[:, 8] == indexes[0][ind] * NUMCOLUMN + indexes[1][
                                                ind]]  # points in covered pixel
                                        scene_pcl = scene_pcl[
                                            scene_pcl[:, 8] != indexes[0][ind] * NUMCOLUMN + indexes[1][
                                                ind]]  # all without covered pixel
                                        visible_part = sample_pcl[
                                            sample_pcl[:, 8] == indexes[0][ind] * NUMCOLUMN + indexes[1][
                                                ind]]  # sample points in pixel
                                        if first_part:
                                            first_part = False
                                            visible_sample = visible_part
                                            covered_scene = covered_part
                                        else:
                                            visible_sample = np.append(visible_sample, visible_part, axis=0)
                                            covered_scene = np.append(covered_scene, covered_part, axis=0)

                                    if DEBUG and False:
                                        tmp = copy.deepcopy(scene_pcl)
                                        tmp[:, 7] = 0
                                        tmp = np.append(tmp, visible_sample, axis=0)
                                        visualization(tmp)
                                        tmp = cutout_frame(tmp, img_file, calib_file)
                                        visualization(tmp)
                                        tmp = copy.deepcopy(scene_pcl)
                                        tmp[:, 7] = 1
                                        _, tmp_label, _ = geometrical_front_view(tmp, NUMROW, NUMCOLUMN, max_elevation, min_elevation)
                                        for r in range(NUMROW):
                                            for c in range(NUMCOLUMN):
                                                if tmp_label[r][c] == -1:
                                                    tmp_label[r][c] = 0
                                        create_image(tmp_label, 'output/scene_FoV.png')

                                        tmp = copy.deepcopy(visible_sample)
                                        tmp[:, 7] = 1
                                        _, tmp_label, _ = geometrical_front_view(tmp, NUMROW, NUMCOLUMN, max_elevation,
                                                                                 min_elevation)
                                        for r in range(NUMROW):
                                            for c in range(NUMCOLUMN):
                                                if tmp_label[r][c] == -1:
                                                    tmp_label[r][c] = 0
                                        create_image(tmp_label, 'output/sample_FoV.png')

                                    if num_bikes != remaining_bikes:
                                        if len(sample_pcl) - len(visible_sample) + num_point_removed_by_cutout < sample_annotation['removable_points'] * 0.3:     # max 15% is easy
                                            print(f'It would be easy case and we do not like those xD')
                                            print(f'Covered {len(sample_pcl) - len(visible_sample)}, cuted out {num_point_removed_by_cutout}, sum {len(sample_pcl) - len(visible_sample) + num_point_removed_by_cutout}')
                                            random_tmp = sample_annotation['removable_points']
                                            print(f'Removable points {random_tmp} min removed points {random_tmp * 0.3}, pcl len {len(sample_pcl)}')
                                            continue

                                    sample_annotation['removable_points'] = sample_annotation['removable_points'] - (
                                                len(sample_pcl) - len(visible_sample)) - num_point_removed_by_cutout

                                    assert sample_annotation['class'] == inserted_class, f'Inserted class does not match'

                                    if len(visible_sample) == 0:
                                        print(f'{RED}No visible part of {inserted_class}{DEFAULT}')

                                    elif len(visible_sample) < MINIMAL_OBJECT_POINTS:
                                        print(f'It is visible just {YELLOW}{len(visible_sample)}{DEFAULT} points from {inserted_class}. It could be hard case. We try another one.')

                                    elif sample_annotation['removable_points'] < 0:
                                        print(f'Additional object is heavily occluded. From {original_len_of_sample} points is visible just {len(visible_sample)} points ({(len(visible_sample)/original_len_of_sample)*100:.01f}%).')

                                    else:
                                        scene_annotation, ok = check_covers(scene_annotation, covered_scene)
                                        if not ok:
                                            print(f'New objects will be covering objects.')
                                            continue

                                        add_objects_name.write(f'{object_name}\n')
                                        visible_sample[:, 9] = len(scene_annotation)
                                        print(f'It is visible {GREEN}{len(visible_sample)}{DEFAULT} points from {inserted_class}. Moving to another scene')
                                        match_find = True
                                        SAMPLE_TIMEOUT = False
                                        unplaceble = False

                                        some_object_inserted = True

                                        if DEBUG:
                                            tmp = copy.deepcopy(scene_pcl)
                                            tmp[:, 7] = 0
                                            tmp1 = copy.deepcopy(visible_sample)
                                            tmp1[:, 7] = 1
                                            tmp = np.append(tmp, tmp1, axis=0)
                                            visualization(tmp)

                                        scene_pcl = np.append(scene_pcl, visible_sample, axis=0)

                                        if inserted_class == 'Car':
                                            remaining_car -= 1
                                        elif inserted_class == 'Cyclist':
                                            remaining_bikes -= 1
                                        elif inserted_class == 'Pedestrian':
                                            remaining_pedestrians -= 1

                                        print(f'Still {remaining_car} Cars, {remaining_bikes} Cyclists, and {remaining_pedestrians} Pedestrians remaining.')

                                        if remaining_car > 0 or remaining_bikes > 0 or remaining_pedestrians > 0:
                                            additional_annotations_lines.append(create_annotation_line(sample_data['anno'], sample_annotation, sample_rotation))
                                            additional_dict = read_label_line(additional_annotations_lines[-1])
                                            scene_annotation = np.append(scene_annotation, [additional_dict])
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

                                            additional_annotations_lines.append(
                                                create_annotation_line(sample_data['anno'], sample_annotation,
                                                                       sample_rotation))
                                            ### SAVING DATA

                                            save_data(scene_pcl, all_visible_parts, additional_annotations_lines, save_folder, frame_name)
                                            add_objects_name.close()
                                            add_objects_txt_open = False
                                            saved = True

                                            break

                                    if sample_index == len(possible_sample_pcl)-1 and unplaceble:
                                        print(f'Object: {object_name} was added to unplaceble list')
                                        unplaceble_samples.append(object_name)

                            if (s_index == len(sample_directory_list)-1 or s_index == 3*MAX_NUM_TRIES) and not match_find:
                                if inserted_class == 'Car':
                                    remaining_car = 0
                                    print(f'{RED}TOTAL Car insertion time-out.{DEFAULT}')
                                    SAMPLE_TIMEOUT = True
                                elif inserted_class == 'Cyclist':
                                    remaining_bikes = 0
                                    print(f'{RED}TOTAL insertion time-out.{DEFAULT}')
                                    SAMPLE_TIMEOUT = True
                                elif inserted_class == 'Pedestrian':
                                    remaining_pedestrians = 0
                                    print(f'{RED}TOTAL Pedestrian insertion time-out.{DEFAULT}')
                                    SAMPLE_TIMEOUT = True
                                print(f'Still {remaining_car} Cars, {remaining_bikes} Cyclists, and {remaining_pedestrians} Pedestrians remaining.')

                            if s_index == end_index-1 and not match_find:
                                if inserted_class == 'Car':
                                    remaining_car -= 1
                                    print(f'{YELLOW}Car insertion time-out.{DEFAULT}')
                                    SAMPLE_TIMEOUT = True

                                    if remaining_car <= 0:
                                        match_find = True

                                elif inserted_class == 'Cyclist':
                                    remaining_bikes -= 1
                                    print(f'{YELLOW}Bike insertion time-out.{DEFAULT}')
                                    SAMPLE_TIMEOUT = True

                                    if remaining_bikes <= 0:
                                        match_find = True

                                elif inserted_class == 'Pedestrian':
                                    remaining_pedestrians -= 1
                                    print(f'{YELLOW}Pedestrian insertion time-out.{DEFAULT}')
                                    SAMPLE_TIMEOUT = True

                                    if remaining_pedestrians <= 0:
                                        match_find = True

                                print(f'Still {remaining_car} Cars, {remaining_bikes} Cyclists, and {remaining_pedestrians} Pedestrians remaining.')

                                if remaining_car <= 0 and remaining_bikes <= 0 and remaining_pedestrians <= 0 and some_object_inserted:

                                    ### SAVING DATA
                                    save_data(scene_pcl, all_visible_parts, additional_annotations_lines, save_folder, frame_name)
                                    add_objects_name.close()
                                    add_objects_txt_open = False
                                    saved = True
                                    match_find = True

                                quit = False
                                break

                if some_object_inserted and not saved:

                    save_data(scene_pcl, all_visible_parts, additional_annotations_lines, save_folder, frame_name)
                    add_objects_name.close()
                    add_objects_txt_open = False
                    saved = True

                assert not (some_object_inserted and not saved), f'Some object was inserted but data was not saved.'
