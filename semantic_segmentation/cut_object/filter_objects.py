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
                    sequence = f'{sequence:02d}'

            dataset_functions = SemanticKITTI(config, sequence)

    elif dataset == 'Waymo':
        print('Waymo was chosen')

        with open('../config/waymo.yaml', 'r') as file:
            config = yaml.safe_load(file)
            dataset_functions = Waymo(config)

    return config, dataset_functions, sequence


if __name__ == '__main__':

    config, _, _ = dataset_selection()

    save_path = config['path']['bbox_path']
    data_path = config['path']['dataset_path']
    anno_path = config['path']['annotation_path']
    classes = config['insertion']['classes']

    assert os.path.exists(f'{save_path}'), f'Root folder does not exist'

    for c in classes:
        cl = config['labels'][c]
        assert os.path.exists(f'{save_path}/{cl}'), f'{cl} folder does not exist'

    for c in classes:
        cl = config['labels'][c]
        print(cl)
        for i in range(100):
            print(f'\r {i:3d}', end='')
            sample_adresses = glob.glob(f'{save_path}/{cl}/*_{i:03d}_m.npz')

            if len(sample_adresses) == 0:
                continue

            sum = np.zeros(360)
            number = np.zeros(360)
            for sample in sample_adresses:

                sample_data = np.load(sample, allow_pickle=True)

                rotation = int(np.rad2deg(float(str(sample_data['anno']).split(' ')[7])) + 180)

                sum[rotation] += len(sample_data['pcl'])
                number[rotation] += 1

            avg = np.where(number != 0, sum/number, np.inf)

            for sample in sample_adresses:

                sample_data = np.load(sample, allow_pickle=True)

                rotation = int(np.rad2deg(float(str(sample_data['anno']).split(' ')[7])) + 180)

                if avg[rotation] > len(sample_data['pcl']):
                    os.remove(f'{sample}')



