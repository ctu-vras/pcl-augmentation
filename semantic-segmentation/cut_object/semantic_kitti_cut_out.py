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

    if not os.path.exists(f'{save_path}'):
        os.mkdir(f'{save_path}')

    for c in classes:
        class_name = config['labels'][c]
        if not os.path.exists(f'{save_path}/{class_name}'):
            os.mkdir(f'{save_path}/{class_name}')

    for sequence in config['split']['train']:

        current_directory = glob.glob(f'{data_path}/sequences/{sequence:02d}/velodyne/*.bin')
        current_directory.sort()

        for vel_path in current_directory:
            tmp = vel_path.split('/')
            tmp[-2] = 'labels'
            sem_path = '/'.join(tmp)
            sem_path = sem_path.split('.')
            sem_path[-1] = 'label'
            sem_path = '.'.join(sem_path)
            frame = tmp[-1].split('.')[0]
            print(f'{sequence:02d}/{frame}')
            points = np.fromfile(f'{vel_path}', dtype=np.float32).reshape(-1, 4)

            semantic_labels = np.fromfile(f'{sem_path}', dtype=np.uint32).reshape(-1, 1)
            semantic_labels = semantic_labels & 0xFFFF

            points = np.hstack((points, semantic_labels))

            bboxes = open(f'{anno_path}/sequences/{sequence:02d}/bbox/{frame}.txt', 'r')

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

                if MAC:
                    r = R.from_matrix(rot_matrix)
                else:
                    r = R.from_dcm(rot_matrix)

                quaternions = r.as_quat()

                save_annotation = [[sample_x, sample_y, sample_z],
                                   [quaternions[0], quaternions[1], quaternions[2], quaternions[3]],
                                   [sample_width, sample_lenght, sample_height]]

                my_annotation = make_dictionary(save_annotation)

                bounding_box = cut_bounding_box(points, my_annotation)

                if int(annotation_items[0]) == 253:      # moving bicyclist
                    bicyclist = bounding_box[bounding_box[:, 4] == 253]
                    bike = bounding_box[bounding_box[:, 4] == 11]
                    bounding_box = np.array([])
                    if len(bicyclist) > 0:
                        bounding_box = bicyclist
                    if len(bike) > 0:
                        if len(bounding_box) > 0:
                            bounding_box = np.append(bounding_box, bike, axis=0)
                        else:
                            bounding_box = bike

                elif int(annotation_items[0]) == 255:    # moving motorcyclist
                    motocyclist = bounding_box[bounding_box[:, 4] == 255]
                    bike = bounding_box[bounding_box[:, 4] == 15]
                    bounding_box = np.array([])
                    if len(motocyclist) > 0:
                        bounding_box = motocyclist
                    if len(bike) > 0:
                        if len(bounding_box) > 0:
                            bounding_box = np.append(bounding_box, bike, axis=0)
                        else:
                            bounding_box = bike
                else:
                    bounding_box = bounding_box[bounding_box[:, 4] == int(annotation_items[0])]

                if len(bounding_box) < config['insertion']['min_points'][int(annotation_items[0])]:
                    continue

                shortcut = config['insertion']['labels_shortcut'][int(annotation_items[0])]

                np.savez(f'{save_path}{save_folder}{shortcut}{sequence:02d}-{frame}_{int(classes_count[classes.index(int(annotation_items[0]))]):02d}_'
                             f'{int(np.sqrt(sample_x ** 2 + sample_y ** 2)):03d}_m', anno=annotation, pcl=bounding_box)

