import numpy as np
import os
from PIL import Image
import math
import glob
import yaml
from datasets import *

RGB_CLASS = np.array([[0, 0, 128], [0, 191, 255], [255, 0, 255], [123, 123, 123], [0, 255, 0], [0, 255, 0], [0, 0, 255]])


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


def create_matrix(transform_data):
    out = np.zeros((4, 4))
    out[3][3] = 1

    for row in range(0, 3):
        for column in range(0, 4):
            out[row][column] = float(transform_data[row * 4 + column])

    return out


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
    '''
    making driveable area from scenes
    '''
    config, dataset_functions, sequence = dataset_selection()

    data_path = config['path']['dataset_path']
    save_path = config['path']['maps_path']
    save_path = save_path.split('/')
    save_path.pop()
    save_path.pop()
    save_path.pop()
    save_path = '/'.join(save_path)

    surface_labels = config['insertion']['placement_labels'][1] + config['insertion']['placement_labels'][2] + \
                     config['insertion']['placement_labels'][3]

    if not os.path.exists(f'{save_path}/maps'):
        os.mkdir(f'{save_path}/maps')
    if not os.path.exists(f'{save_path}/maps/small'):
        os.mkdir(f'{save_path}/maps/small')
    if not os.path.exists(f'{save_path}/maps/small/picture'):
        os.mkdir(f'{save_path}/maps/small/picture')
    if not os.path.exists(f'{save_path}/maps/small/npz'):
        os.mkdir(f'{save_path}/maps/small/npz')

    old_sequence = dataset_functions.sequence
    new_sequence = dataset_functions.sequence
    idx = 0
    num_seq_pcl = 0

    while len(dataset_functions) > 0:
        max_global_x = -math.inf
        min_global_x = math.inf
        max_global_y = -math.inf
        min_global_y = math.inf
        print(f'OLD:{old_sequence} CREATING NEW:{new_sequence}')
        while old_sequence == new_sequence:

            pcl, t_matrix, _, _, new_sequence = dataset_functions[idx]

            if old_sequence != new_sequence:
                break

            points = pcl[:, :4]
            points[:, 3] = 1
            points = t_matrix @ points.T
            points = (points / points[3, :]).T

            if np.max(points, axis=0)[0] > max_global_x:
                max_global_x = np.max(points, axis=0)[0]
            if np.min(points, axis=0)[0] < min_global_x:
                min_global_x = np.min(points, axis=0)[0]
            if np.max(points, axis=0)[1] > max_global_y:
                max_global_y = np.max(points, axis=0)[1]
            if np.min(points, axis=0)[1] < min_global_y:
                min_global_y = np.min(points, axis=0)[1]

            idx += 1
            if idx == len(dataset_functions):
                break

        num_seq_pcl = idx

        print(f'SEQUENCE {old_sequence}')
        print('LEFT TOP: [', min_global_x, ',', min_global_y, ']')
        print('RIGHT BOTTOM: [', max_global_x, ',', max_global_y, ']')

        min_global_x = int(np.floor(min_global_x))
        min_global_y = int(np.floor(min_global_y))

        max_global_y = int(max_global_y) + 1
        max_global_x = int(max_global_x) + 1

        size_x = int(max_global_x - min_global_x)

        size_y = int(max_global_y - min_global_y)

        print(size_x, size_y)

        scene_driveable_area = np.zeros((size_x, size_y))

        skip = False

        for i in range(num_seq_pcl):
            pcl, t_matrix, _, _, _ = dataset_functions[0]
            dataset_functions.delete_item(0, subdirectoties=False)

            if os.path.exists(f'{save_path}/maps/small/picture/{old_sequence}.png'):
                skip = True
                continue

            points = pcl[:, :4]
            points[:, 3] = 1
            points = t_matrix @ points.T
            points = (points / points[3, :]).T
            pcl[:, :3] = points[:, :3]

            for point in pcl:

                if not point[4] in surface_labels:
                    continue

                position_x = point[0] - min_global_x
                position_y = point[1] - min_global_y
                assert not (position_x < 0 or position_y < 0), f'Indexing error in: {old_sequence}/{i}, position x:{position_x}, y:{position_y}, {point}'

                if point[4] in config['insertion']['placement_labels'][1]:
                    if scene_driveable_area[int(position_x)][int(position_y)] != 3:
                        scene_driveable_area[int(position_x)][int(position_y)] = 1

                elif point[4] in config['insertion']['placement_labels'][3]:
                    scene_driveable_area[int(position_x)][int(position_y)] = 3

                else:
                    if scene_driveable_area[int(position_x)][int(position_y)] != 3:
                        scene_driveable_area[int(position_x)][int(position_y)] = 2

        if not skip:

            create_image(scene_driveable_area, f'{save_path}/maps/small/picture/{old_sequence}.png')
            np.savez(f'{save_path}/maps/small/npz/{old_sequence}',
                     move=np.array([[min_global_x], [min_global_y], [0], [1]]), map=scene_driveable_area)

        idx = 0
        old_sequence = new_sequence
