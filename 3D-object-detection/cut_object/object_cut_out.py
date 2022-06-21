import os
import numpy as np
import copy
import math
import glob
import yaml
from scipy.spatial.transform import Rotation as R

RGB_CLASS = np.array([[0, 0, 128], [0, 191, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]])
TEST = False
TEST_1 = True
MAC = False

from cut_bbox import cut_bounding_box
from cutout import *
from datasets import KITTI


def make_dictionary(annotation_array):
    center = {'x': annotation_array[0][0], 'y': annotation_array[0][1], 'z': annotation_array[0][2]}
    rotation = {'x': annotation_array[1][0], 'y': annotation_array[1][1], 'z': annotation_array[1][2], 'w': annotation_array[1][3]}
    annotation_dictionary = {'center': center, 'rotation': rotation, 'length': annotation_array[2][0],
                             'width': annotation_array[2][1], 'height': annotation_array[2][2]}
    return annotation_dictionary


def dataset_selection():
    quit = False
    dataset = 'KITTI'
    while not quit:
        print('Choose dataset:')
        print('1 - KITTI')
        tmp = '1'
        if tmp == '1' or tmp == '2':
            quit = True
            if tmp == '1':
                dataset = 'KITTI'
            else:
                # place for different dataset
                pass
        else:
            print('Wrong input try again')

    if dataset == 'KITTI':
        print('KITTI was chosen')
        if MAC:
            with open('config/KITTI-mac.yaml', 'r') as file:
                config = yaml.safe_load(file)
                dataset_functions = KITTI(config)

        else:
            with open('../config/KITTI.yaml', 'r') as file:
                config = yaml.safe_load(file)

                dataset_functions = KITTI(config)

    return config, dataset_functions


if __name__ == '__main__':

    config, dataset_functions = dataset_selection()

    data_path = config['path']['dataset_path']
    save_path = config['path']['sample_path']

    train_txt_path = config['path']['train_txt_path']
    classes = config['insertion']['classes']

    for c in config['insertion']['classes']:

        if not os.path.exists(f'{save_path}/{c}'):
            os.mkdir(f'{save_path}/{c}')

    while len(dataset_functions) > 0:

        print(f'\rRemaining {len(dataset_functions)} frames', end='')

        points, label_address, _, calib_file, img_file = dataset_functions[0]

        dataset_functions.delete_item(0)

        label = open(f'{label_address}', 'r')
        end_of_file = False

        classes_count = np.zeros(len(classes))

        frame = label_address.split('/')[-1].split('.')[0]

        while not end_of_file:
            annotation = label.readline()

            if len(annotation) == 0:
                end_of_file = True
                continue

            annotation_items = annotation.split(' ')

            sample_class = annotation_items[0]

            if not (sample_class in classes):
                continue

            sample_occluded = int(annotation_items[2])

            if sample_occluded != 0:
                continue

            sample_height = float(annotation_items[8])
            sample_width = float(annotation_items[9])
            sample_lenght = float(annotation_items[10])

            sample_x = float(annotation_items[11])
            sample_y = float(annotation_items[12])
            sample_z = float(annotation_items[13])

            sample_rotation_y = annotation_items[14]

            corrected_x = float(sample_z) + 0.27
            corrected_y = float(sample_x)*-1
            corrected_z = float(sample_y)*-1 - 0.08

            z_rot = float(sample_rotation_y) * -1

            rot_matrix = [[math.cos(z_rot), -1*math.sin(z_rot), 0],
                          [math.sin(z_rot), math.cos(z_rot), 0],
                          [0, 0, 1]]

            r = R.from_dcm(rot_matrix)

            quaternions = r.as_quat()

            save_annotation = [[corrected_x, corrected_y, corrected_z],
                               [quaternions[0], quaternions[1], quaternions[2], quaternions[3]],
                               [sample_width + 0.2, sample_lenght + 0.2, sample_height + 0.1],
                               [sample_occluded]]

            my_annotation = make_dictionary(save_annotation)

            expand_annotation = copy.deepcopy(my_annotation)
            expand_annotation['length'] += 0.2
            expand_annotation['width'] += 0.2
            expand_annotation['height'] += 0.2

            expand_bbox = cut_bounding_box(points, expand_annotation)

            if len(expand_bbox) != len(cutout_frame(expand_bbox, img_file, calib_file)):
                continue

            save_folder = f'/{sample_class}/'

            classes_count[classes.index(sample_class)] += 1

            bounding_box = cut_bounding_box(points, my_annotation)
            bounding_box = bounding_box[bounding_box[:, 4] != config['labels']['Road']]
            bounding_box = bounding_box[bounding_box[:, 4] != config['labels']['Parking']]
            bounding_box = bounding_box[bounding_box[:, 4] != config['labels']['Sidewalk']]

            bounding_box = bounding_box[:, 0:4]

            if len(bounding_box) < config['insertion']['min_points'][sample_class]:
                continue

            label_column = np.ones((len(bounding_box), 1))
            bounding_box = np.hstack((bounding_box, label_column))
            np.savez(f'{save_path}{save_folder}{config["insertion"]["labels_shortcut"][sample_class]}{frame}_'
                     f'{int(classes_count[classes.index(sample_class)])}_'
                     f'{int(np.sqrt(corrected_x ** 2 + corrected_y ** 2))}_m', anno=annotation, pcl=bounding_box)


