import numpy as np
import os
from PIL import Image
import glob
from skimage.util import img_as_ubyte
from skimage.morphology import closing, dilation
from skimage.morphology import disk

RGB_CLASS = np.array([[0, 0, 128], [0, 191, 255], [255, 0, 255], [0, 255, 0], [123, 123, 123], [0, 255, 0], [0, 0, 255]])

DATA_PATH = ''          # path to KITTI dataset

SAVE_DIRECTORY = ''     # path to output directory


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
            if labels_idx == 255:
                labels_idx = 1
            rgb[i][j] = RGB_CLASS[labels_idx]

    rgb = np.uint8(rgb)
    img = Image.fromarray(rgb, 'RGB')
    img.save(filename)


if __name__ == '__main__':
    '''
    making driveable area from scenes
    '''
    if not os.path.exists(f'{SAVE_DIRECTORY}/maps'):
        os.mkdir(f'{SAVE_DIRECTORY}/maps')

    if not os.path.exists(f'{SAVE_DIRECTORY}/maps/road_maps'):
        os.mkdir(f'{SAVE_DIRECTORY}/maps/road_maps')

    if not os.path.exists(f'{SAVE_DIRECTORY}/maps/road_maps/picture'):
        os.mkdir(f'{SAVE_DIRECTORY}/maps/road_maps/picture')

    if not os.path.exists(f'{SAVE_DIRECTORY}/maps/road_maps/npz'):
        os.mkdir(f'{SAVE_DIRECTORY}/maps/road_maps/npz')

    if not os.path.exists(f'{SAVE_DIRECTORY}/maps/pedestrian_area'):
        os.mkdir(f'{SAVE_DIRECTORY}/maps/pedestrian_area')

    if not os.path.exists(f'{SAVE_DIRECTORY}/maps/pedestrian_area/picture'):
        os.mkdir(f'{SAVE_DIRECTORY}/maps/pedestrian_area/picture')

    if not os.path.exists(f'{SAVE_DIRECTORY}/maps/pedestrian_area/npz'):
        os.mkdir(f'{SAVE_DIRECTORY}/maps/pedestrian_area/npz')

    velodyne_directory = glob.glob(f'{DATA_PATH}/velodyne/*.bin')
    velodyne_directory.sort()

    pseudo_label_directory = glob.glob(f'{DATA_PATH}/semantic_labels/labels/*.labels')
    pseudo_label_directory.sort()

    for file in velodyne_directory:
        if file.endswith('.bin'):

            print(file)
            point_cloud = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
            frame_number = int(file.split('/')[-1].split('.')[0])
            labels = np.fromfile(f'{DATA_PATH}/semantic_labels/labels/{frame_number:06d}.label',
                                 dtype=np.uint32).reshape(-1, 1)

            assert len(point_cloud) == len(labels), 'Error in label vs points number.'

            min_x = int(min(point_cloud[:, 0]))
            min_y = int(min(point_cloud[:, 1]))

            max_x = int(max(point_cloud[:, 0])) + 1
            max_y = int(max(point_cloud[:, 1])) + 1

            size_x = int(max_x - min_x)

            size_y = int(max_y - min_y)

            print(size_x, size_y)

            scene_driveable_area = np.zeros((size_x, size_y))
            pedestrian_area = np.zeros((size_x, size_y))

            labels = labels & 0xFFFF

            point_cloud = np.hstack((point_cloud, labels))

            for point in point_cloud:
                if point[4] != 40:
                    continue

                position_x = point[0] - min_x
                position_y = point[1] - min_y

                assert int(position_x) >= 0 or int(position_y) >= 0 or int(position_x) < size_x or int(position_y) < size_y, f'Indexing error in: {file}'

                if point[4] == 40:
                    scene_driveable_area[int(position_x)][int(position_y)] = 1

            mask_phantom = img_as_ubyte(scene_driveable_area)
            selem = disk(4)
            closed_road = closing(mask_phantom, selem)
            for r in range(size_x):
                for c in range(size_y):
                    if closed_road[r][c] != 0:
                        closed_road[r][c] = 1

            create_image(closed_road, f'{SAVE_DIRECTORY}/maps/road_maps/picture/{frame_number:06d}-closed-disk4.png')
            np.savez(f'{SAVE_DIRECTORY}/maps/road_maps/npz/{frame_number:06d}', map=closed_road, min_x=min_x,
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

            create_image(dilated_pedestrian_area, f'{SAVE_DIRECTORY}/maps/pedestrian_area/picture/{frame_number:06d}-dilation-disk2.png')
            np.savez(f'{SAVE_DIRECTORY}/maps/pedestrian_area/npz/{frame_number:06d}', map=dilated_pedestrian_area, min_x=min_x,
                     min_y=min_y)