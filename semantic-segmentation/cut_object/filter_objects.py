import os
import numpy as np
import copy
import math
import glob
import yaml
from scipy.spatial.transform import Rotation as R

from tools.cut_bbox import cut_bounding_box

MAC = False
if MAC:
    from tools.cut_bbox import separate_bbox
    from tools.visualization import *


def make_dictionary(annotation_array):
    center = {'x': annotation_array[0][0], 'y': annotation_array[0][1], 'z': annotation_array[0][2]}
    rotation = {'x': annotation_array[1][0], 'y': annotation_array[1][1], 'z': annotation_array[1][2], 'w': annotation_array[1][3]}
    annotation_dictionary = {'center': center, 'rotation': rotation, 'length': annotation_array[2][0],
                             'width': annotation_array[2][1], 'height': annotation_array[2][2]}
    return annotation_dictionary


if __name__ == '__main__':

    if not os.path.exists('../config/semantic-kitti-mac.yaml'):
        MAC = False

    if MAC:
        with open('../config/semantic-kitti-mac.yaml', 'r') as file:
            config = yaml.safe_load(file)

    else:
        with open('../config/semantic-kitti.yaml', 'r') as file:
            config = yaml.safe_load(file)

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
        for i in range(100):
            sample_adresses = glob.glob(f'{save_path}/{cl}/*_{i:03d}_m.npz')

            if len(sample_adresses) == 0:
                continue

            sum = 0
            for sample in sample_adresses:

                sample_data = np.load(sample, allow_pickle=True)

                sum += len(sample_data['pcl'])

            avg = sum/len(sample_adresses)

            for sample in sample_adresses:

                sample_data = np.load(sample, allow_pickle=True)

                if avg > len(sample_data['pcl']):
                    os.remove(f'{sample}')


