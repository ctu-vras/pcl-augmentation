import numpy as np
import os
from PIL import Image
import math
import glob
# import open3d as o3d
import yaml

MAC = False

RGB_CLASS = np.array([[0, 0, 128], [0, 191, 255], [255, 0, 255], [123, 123, 123], [0, 255, 0], [0, 255, 0], [0, 0, 255]])

VELO_2_CAM = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                       [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                       [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
                       [0, 0, 0, 1]])
MY_CALIB = np.array([[0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]])

ANNOTATION = False


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


if __name__ == '__main__':
    '''
    making driveable area from scenes
    '''
    if MAC:
        with open('../config/semantic-kitti-mac.yaml', 'r') as file:
            config = yaml.safe_load(file)

    else:
        with open('../config/semantic-kitti.yaml', 'r') as file:
            config = yaml.safe_load(file)

    data_path = config['path']['dataset_path']
    save_path = config['path']['maps_path']
    save_path = save_path.split('/')
    save_path.pop()
    save_path.pop()
    save_path.pop()
    save_path = '/'.join(save_path)


    if not os.path.exists(f'{save_path}/maps'):
        os.mkdir(f'{save_path}/maps')
    if not os.path.exists(f'{save_path}/maps/small'):
        os.mkdir(f'{save_path}/maps/small')
    if not os.path.exists(f'{save_path}/maps/small/picture'):
        os.mkdir(f'{save_path}/maps/small/picture')
    if not os.path.exists(f'{save_path}/maps/small/npz'):
        os.mkdir(f'{save_path}/maps/small/npz')

    for sequence in config['split']['train']:
        current_directory = glob.glob(f'{data_path}/sequences/{sequence:02d}/velodyne/*.bin')
        current_directory.sort()

        poses = open(f'{data_path}/sequences/{sequence:02d}/poses.txt')

        max_global_x = -math.inf
        min_global_x = math.inf
        max_global_y = -math.inf
        min_global_y = math.inf

        for file in current_directory:
            if file.endswith('.bin'):
                print(file)
                point_cloud = np.fromfile(file, dtype=np.float32).reshape(-1, 4)

                transformation_matrix = poses.readline()
                transformation_matrix = transformation_matrix.split(' ')

                assert len(transformation_matrix) == 12, 'Error in poses. Point-cloud has no transformation matrix.'

                transformation_matrix = create_matrix(transformation_matrix)

                transformation_matrix = np.dot(transformation_matrix, VELO_2_CAM)
                transformation_matrix = np.dot(np.linalg.inv(MY_CALIB), transformation_matrix)

                for i in range(len(point_cloud)):
                    point = point_cloud[i]
                    x = np.array([[point[0]], [point[1]], [point[2]], [1.]])
                    global_position = np.dot(transformation_matrix, x)

                    if global_position[0][0] > max_global_x:
                        max_global_x = global_position[0][0]
                    if global_position[0][0] < min_global_x:
                        min_global_x = global_position[0][0]
                    if global_position[1][0] > max_global_y:
                        max_global_y = global_position[1][0]
                    if global_position[1][0] < min_global_y:
                        min_global_y = global_position[1][0]
        print(f'SEQUENCE {sequence:02d}')
        print('LEFT TOP: [', min_global_x, ',', min_global_y, ']')
        print('RIGHT BOTTOM: [', max_global_x, ',', max_global_y, ']')

        min_global_x = int(min_global_x)
        min_global_y = int(min_global_y)

        max_global_y = int(max_global_y) + 1
        max_global_x = int(max_global_x) + 1

        size_x = int(max_global_x-min_global_x)

        size_y = int(max_global_y-min_global_y)

        print(size_x, size_y)

        scene_driveable_area = np.zeros((size_x, size_y))

        poses.close()

        poses = open(f'{data_path}/sequences/{sequence:02d}/poses.txt')

        for file in current_directory:
            if file.endswith('.bin'):

                print(file)
                point_cloud = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
                frame_number = int(file.split('/')[-1].split('.')[0])
                labels = np.fromfile(f'{data_path}/sequences/{sequence:02d}/labels/{frame_number:06d}.label', dtype=np.uint32).reshape(-1,1)

                labels = labels & 0xFFFF

                point_cloud = np.hstack((point_cloud, labels))

                transformation_matrix = poses.readline()
                transformation_matrix = transformation_matrix.split(' ')
                assert len(transformation_matrix) == 12, 'Error in poses. Point-cloud has no transformation matrix.'

                transformation_matrix = create_matrix(transformation_matrix)

                transformation_matrix = np.dot(transformation_matrix, VELO_2_CAM)
                transformation_matrix = np.dot(np.linalg.inv(MY_CALIB), transformation_matrix)

                for i in range(len(point_cloud)):
                    point = point_cloud[i]
                    if point[4] != 40 and point[4] != 48 and point[4] != 44:
                        continue
                    x = np.array([[point[0]], [point[1]], [point[2]], [1.]])
                    global_position = np.dot(transformation_matrix, x)

                    position_x = global_position[0][0] - min_global_x
                    position_y = global_position[1][0] - min_global_y
                    assert not (position_x < 0 or position_y < 0), f'Indexing error in: {file}'

                    if point[4] == 40:
                        scene_driveable_area[int(position_x)][int(position_y)] = 1
                    elif point[4] == 44:
                        scene_driveable_area[int(position_x)][int(position_y)] = 3
                    else:
                        scene_driveable_area[int(position_x)][int(position_y)] = 2

        poses.close()

        create_image(scene_driveable_area, f'{save_path}/maps/small/picture/{sequence:02d}.png')
        np.savez(f'{save_path}/maps/small/npz/{sequence:02d}', move=np.array([[min_global_x], [min_global_y], [0], [1]]), map=scene_driveable_area)