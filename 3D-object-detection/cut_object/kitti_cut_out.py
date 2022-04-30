import os
import numpy as np
import copy
import math
import glob
from scipy.spatial.transform import Rotation as R

RGB_CLASS = np.array([[0, 0, 128], [0, 191, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]])
TEST = False

from cut_bbox import cut_bounding_box
from cutout import *
# from cut_bbox import visualization

data_path = '/datagrid/personal/vacekpa2/students/kitti/training_default'
save_path = '/datagrid/personal/vacekpa2/students/sebekpe1/KITTI/test_objects/'
TRAIN_TXT_PATH = '/datagrid/personal/vacekpa2/students/kitti/training_default/train.txt'


def make_dictionary(annotation_array):
    center = {'x': annotation_array[0][0], 'y': annotation_array[0][1], 'z': annotation_array[0][2]}
    rotation = {'x': annotation_array[1][0], 'y': annotation_array[1][1], 'z': annotation_array[1][2], 'w': annotation_array[1][3]}
    annotation_dictionary = {'center': center, 'rotation': rotation, 'length': annotation_array[2][0],
                             'width': annotation_array[2][1], 'height': annotation_array[2][2]}
    return annotation_dictionary


if __name__ == '__main__':
    if not os.path.exists(save_path + 'kitti_pedestrians'):
        os.mkdir(save_path + 'kitti_pedestrians')

    if not os.path.exists(save_path + 'kitti_bikes'):
        os.mkdir(save_path + 'kitti_bikes')

    if not os.path.exists(save_path + 'kitti_cars'):
        os.mkdir(save_path + 'kitti_cars')

    quit = False

    train_txt = open(TRAIN_TXT_PATH)
    current_directory = []
    while not quit:
        frame_name = train_txt.readline()

        if len(frame_name) == 0:
            train_txt.close()
            quit = True

        else:
            frame_name = int(frame_name)
            current_directory.append(frame_name)
            

    for frame in current_directory:
        print(f'{frame:06d}')
        points = np.fromfile(f'{data_path}/velodyne/{frame:06d}.bin', dtype=np.float32).reshape(-1, 4)
        semantic_labels = np.fromfile(f'{data_path}/semantic_labels/labels/{frame:06d}.label',
                                      dtype=np.uint32).reshape(-1, 1)
        semantic_labels = semantic_labels & 0xFFFF
        points = np.hstack((points, semantic_labels))
        label = open(f'{data_path}/label_2/{frame:06d}.txt', 'r')
        calib_file = f'{data_path}/calib/{frame:06d}.txt'
        img_file = f'{data_path}/image_2/{frame:06d}.png'
        end_of_file = False
        bikes_count = 0
        pedestrian_count = 0
        vehicles_count = 0

        while not end_of_file:
            annotation = label.readline()

            if len(annotation) == 0:
                end_of_file = True
                continue

            annotation_items = annotation.split(' ')

            sample_class = annotation_items[0]

            if sample_class == 'Pedestrian':
                save_folder = 'kitti_pedestrians/'
                pedestrian_count += 1

            elif sample_class == 'Cyclist':
                save_folder = 'kitti_bikes/'
                bikes_count += 1

            elif sample_class == 'Car':
                save_folder = 'kitti_cars/'
                vehicles_count += 1

            else:
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

            expand_bbox = cut_bounding_box(points, my_annotation, expand=True)

            if len(expand_bbox) != len(cutout_frame(expand_bbox, img_file, calib_file)):
                # visualization(expand_bbox, colors=False)
                # visualization(cutout_frame(expand_bbox, img_file, calib_file), colors=False)
                continue

            bounding_box = cut_bounding_box(points, my_annotation)
            bounding_box = bounding_box[bounding_box[:, 4] != 40]
            bounding_box = bounding_box[bounding_box[:, 4] != 44]
            bounding_box = bounding_box[bounding_box[:, 4] != 48]

            bounding_box = bounding_box[:, 0:4]

            if sample_class == 'Pedestrian':
                label_column = np.ones((len(bounding_box), 1)) * 6
                bounding_box = np.hstack((bounding_box, label_column))
                np.savez(f'{save_path}{save_folder}P{frame:06d}_{pedestrian_count}_'
                         f'{int(np.sqrt(corrected_x ** 2 + corrected_y ** 2))}_m', anno=annotation, pcl=bounding_box)
            elif sample_class == 'Cyclist':
                label_column = np.ones((len(bounding_box), 1)) * 7
                bounding_box = np.hstack((bounding_box, label_column))
                np.savez(f'{save_path}{save_folder}B{frame:06d}_{bikes_count}_'
                         f'{int(np.sqrt(corrected_x ** 2 + corrected_y ** 2))}_m', anno=annotation, pcl=bounding_box)
            elif sample_class == 'Car':
                label_column = np.ones((len(bounding_box), 1))
                bounding_box = np.hstack((bounding_box, label_column))
                np.savez(f'{save_path}{save_folder}V{frame:06d}_{vehicles_count}_'
                         f'{int(np.sqrt(corrected_x ** 2 + corrected_y ** 2))}_m', anno=annotation, pcl=bounding_box)

