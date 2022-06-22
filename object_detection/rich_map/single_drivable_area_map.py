import numpy as np
import os
from PIL import Image
import glob
import yaml
from skimage.util import img_as_ubyte
from skimage.morphology import closing, dilation
from skimage.morphology import disk
from object_detection.Real3DAug.tools.datasets import *

RGB_CLASS = np.array([[0, 0, 128], [0, 191, 255], [255, 0, 255], [0, 255, 0], [123, 123, 123], [0, 255, 0], [0, 0, 255]])

DEBUG = False

MAC = False


def create_image(labels, filename, road=True):
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
            if not road and labels_idx == 1:
                labels_idx = 2
            if labels_idx == 255:
                labels_idx = 1
            rgb[i][j] = RGB_CLASS[labels_idx]

    rgb = np.uint8(rgb)
    img = Image.fromarray(rgb, 'RGB')
    img.save(filename)


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
    '''
    making driveable area from scenes
    '''
    config, dataset_functions = dataset_selection()

    data_path = config['path']['dataset_path']
    save_path = config['path']['maps_path']

    if not os.path.exists(f'{save_path}/maps'):
        os.mkdir(f'{save_path}/maps')

    if not os.path.exists(f'{save_path}/maps/road_maps'):
        os.mkdir(f'{save_path}/maps/road_maps')

    if not os.path.exists(f'{save_path}/maps/road_maps/npz'):
        os.mkdir(f'{save_path}/maps/road_maps/npz')

    if not os.path.exists(f'{save_path}/maps/pedestrian_area'):
        os.mkdir(f'{save_path}/maps/pedestrian_area')

    if not os.path.exists(f'{save_path}/maps/pedestrian_area/npz'):
        os.mkdir(f'{save_path}/maps/pedestrian_area/npz')

    if DEBUG:
        if not os.path.exists(f'{save_path}/maps/raw'):
            os.mkdir(f'{save_path}/maps/raw')

        if not os.path.exists(f'{save_path}/maps/raw/picture'):
            os.mkdir(f'{save_path}/maps/raw/picture')

        if not os.path.exists(f'{save_path}/maps/road_maps/picture'):
            os.mkdir(f'{save_path}/maps/road_maps/picture')

        if not os.path.exists(f'{save_path}/maps/pedestrian_area/picture'):
            os.mkdir(f'{save_path}/maps/pedestrian_area/picture')

    while len(dataset_functions) > 0:

        print(f'\rRemaining {len(dataset_functions)} frames', end='')

        point_cloud, label_address, _, _, _ = dataset_functions[0]

        frame_number = int(label_address.split('/')[-1].split('.')[0])

        dataset_functions.delete_item(0)

        min_x = int(min(point_cloud[:, 0]))
        min_y = int(min(point_cloud[:, 1]))

        max_x = int(max(point_cloud[:, 0])) + 1
        max_y = int(max(point_cloud[:, 1])) + 1

        size_x = int(max_x - min_x)

        size_y = int(max_y - min_y)

        scene_driveable_area = np.zeros((size_x, size_y))
        pedestrian_area = np.zeros((size_x, size_y))

        for point in point_cloud:
            if point[4] != config['labels']['Road']:
                continue

            position_x = point[0] - min_x
            position_y = point[1] - min_y

            assert int(position_x) >= 0 or int(position_y) >= 0 or int(position_x) < size_x or int(position_y) < size_y, f'Indexing error in: {file}'

            scene_driveable_area[int(position_x)][int(position_y)] = 1

        if DEBUG:
            create_image(scene_driveable_area, f'{save_path}/maps/raw/picture/{frame_number:06d}.png')

        mask_phantom = img_as_ubyte(scene_driveable_area)
        selem = disk(4)
        closed_road = closing(mask_phantom, selem)
        for r in range(size_x):
            for c in range(size_y):
                if closed_road[r][c] != 0:
                    closed_road[r][c] = 1

        if DEBUG:
            create_image(closed_road, f'{save_path}/maps/road_maps/picture/{frame_number:06d}-closed-disk4.png')

        np.savez(f'{save_path}/maps/road_maps/npz/{frame_number:06d}', map=closed_road, min_x=min_x,
                 min_y=min_y)

        for r in range(size_x):
            for c in range(size_y):

                if closed_road[r][c] == 0:

                    near_road = False

                    for d_r in range(-1, 2):
                        for d_c in range(-1, 2):
                            if 0 <= r + d_r < size_x and 0 <= c + d_c < size_y and closed_road[r + d_r][c + d_c] == 1:
                                near_road = True
                                break
                        if near_road:
                            break

                    if near_road:
                        pedestrian_area[r][c] = 1

        mask_phantom = img_as_ubyte(pedestrian_area)
        selem = disk(2)
        dilated_pedestrian_area = dilation(mask_phantom, selem)
        for r in range(size_x):
            for c in range(size_y):
                if dilated_pedestrian_area[r][c] != 0:
                    dilated_pedestrian_area[r][c] = 1

        if DEBUG:
            create_image(dilated_pedestrian_area, f'{save_path}/maps/pedestrian_area/picture/{frame_number:06d}-dilation-disk2.png', road=False)

        np.savez(f'{save_path}/maps/pedestrian_area/npz/{frame_number:06d}', map=dilated_pedestrian_area, min_x=min_x,
                 min_y=min_y)