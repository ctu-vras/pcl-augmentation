import os
import numpy as np
import copy
import math
import glob
import yaml
from scipy.spatial.transform import Rotation as R
from semantic_segmentation.Real3DAug.tools.datasets import *
from tools.cut_bbox import cut_bounding_box

DEBUG = False
if DEBUG:
    from tools.cut_bbox import separate_bbox
    from tools.visualization import *


def make_dictionary(annotation_array):
    center = {'x': annotation_array[0][0], 'y': annotation_array[0][1], 'z': annotation_array[0][2]}
    rotation = {'x': annotation_array[1][0], 'y': annotation_array[1][1], 'z': annotation_array[1][2], 'w': annotation_array[1][3]}
    annotation_dictionary = {'center': center, 'rotation': rotation, 'length': annotation_array[2][0],
                             'width': annotation_array[2][1], 'height': annotation_array[2][2]}
    return annotation_dictionary


def dataset_selection():
    quit = False
    dataset = 'SemanticKITTI'
    while not quit:
        print('Choose pseudo_labels:')
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
        
        with open('../config/semantic-kitti.yaml', 'r') as file:
            config = yaml.safe_load(file)

            quit = False
            while not quit:
                print('Choose sequence')
                sequence = input()
                if int(sequence) in config['split']['train']:
                    quit = True
                    sequence = str(sequences).zfill(2) # f'{sequence:02d}'

            dataset_functions = SemanticKITTI(config, sequence)

    elif dataset == 'Waymo':
        print('Waymo was chosen')

        with open('../config/waymo.yaml', 'r') as file:
            config = yaml.safe_load(file)
            dataset_functions = Waymo(config)

    return config, dataset_functions, sequence


if __name__ == '__main__':

    config, dataset_functions, sequence = dataset_selection()

    save_path = config['path']['bbox_path']
    data_path = config['path']['dataset_path']
    anno_path = config['path']['annotation_path']
    classes = config['insertion']['classes']

    if not os.path.exists(f'{save_path}'):
        os.mkdir(f'{save_path}')

    for c in classes:
        class_name = config['labels'][c]
        if not os.path.exists(f'{save_path}/{class_name}'):
            os.mkdir(f'{save_path}/{class_name}')

    while len(dataset_functions) > 0:

        print(f'\rRemaining {len(dataset_functions):5d} frames', end='')

        points, _, anno_file, _, sequence = dataset_functions[0]

        dataset_functions.delete_item(0)

        if not os.path.exists(anno_file):
            continue

        bboxes = open(f'{anno_file}', 'r')

        end_of_file = False

        classes_count = np.zeros(len(classes))

        while not end_of_file:
            annotation = bboxes.readline()

            if len(annotation) == 0:
                end_of_file = True
                continue

            annotation_items = annotation.split(' ')

            if not (int(annotation_items[0]) in classes):
                continue

            sample_class = config['labels'][int(annotation_items[0])]

            save_folder = f'/{sample_class}/'

            classes_count[classes.index(int(annotation_items[0]))] += 1

            sample_x = float(annotation_items[1])
            sample_y = float(annotation_items[2])
            sample_z = float(annotation_items[3])

            sample_height = float(annotation_items[4])
            sample_width = float(annotation_items[6])
            sample_lenght = float(annotation_items[5])

            sample_rotation_z = float(annotation_items[7])

            rot_matrix = [[math.cos(sample_rotation_z), -1*math.sin(sample_rotation_z), 0],
                          [math.sin(sample_rotation_z), math.cos(sample_rotation_z), 0],
                          [0, 0, 1]]

            r = R.from_dcm(rot_matrix)

            quaternions = r.as_quat()

            save_annotation = [[sample_x, sample_y, sample_z],
                               [quaternions[0], quaternions[1], quaternions[2], quaternions[3]],
                               [sample_width, sample_lenght, sample_height]]

            my_annotation = make_dictionary(save_annotation)

            bounding_box = cut_bounding_box(points, my_annotation)

            bounding_box = bounding_box[bounding_box[:, 4] == int(annotation_items[0])]

            if len(bounding_box) < config['insertion']['min_points'][int(annotation_items[0])]:
                continue

            shortcut = config['insertion']['labels_shortcut'][int(annotation_items[0])]

            frame = anno_file.split('/')[-1].split('.')[0]

            np.savez(f'{save_path}{save_folder}{shortcut}{sequence}-{frame}_{int(classes_count[classes.index(int(annotation_items[0]))]):02d}_'
                         f'{int(np.sqrt(sample_x ** 2 + sample_y ** 2)):03d}_m', anno=annotation, pcl=bounding_box)

    print('Done')

