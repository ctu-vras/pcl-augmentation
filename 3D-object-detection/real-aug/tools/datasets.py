import numpy as np
import glob
import os


def create_read_me(save_folder, config):
    txt = open(f'{save_folder}/setting.txt', 'w')
    txt.write(f'Inserted classes:\n')

    if config['insertion']['random']:
        for c in config['insertion']['classes']:
            txt.write('     ' + c + '\n')
        txt.write('Randomly inserted '+ str(config['insertion']['number_of_object']) + ' objects\n')
    else:
        for c in range(len(config['insertion']['classes'])):
            txt.write('     '+ str(config['insertion']['number_of_classes'][c])+'x   ' + config['insertion']['classes'][c] + '\n')
    txt.close()


def create_annotation(old_address, new_address, additional_annotations_lines):
    old_txt = open(old_address, 'r')
    new_txt = open(new_address, 'w')

    end_of_file = False
    while not end_of_file:
        annotation = old_txt.readline()

        if len(annotation) == 0:
            end_of_file = True
            for a in additional_annotations_lines:
                new_txt.write(a)
            continue

        new_txt.write(annotation)

    old_txt.close()
    new_txt.close()


class KITTI():
    def __init__(self, config):

        self.config = config
        self.data_path = config['path']['dataset_path']
        self.label_path = config['path']['label_path']
        self.train_txt_path = config['path']['train_txt_path']
        self.save_output_folder = config['path']['output_path']

        self.velodyne_list = np.array([])

        self.create_velodyne_list()

    def __len__(self):
        return len(self.velodyne_list)

    def __getitem__(self, idx):

        file = self.velodyne_list[idx]

        frame_name = file.split('/')[-1].split('.')[0]

        pcl = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(f'{self.label_path}/{frame_name}.label', dtype=np.uint32).reshape(-1, 1)
        instance = labels >> 16
        semantic_labels = labels & 0xFFFF
        pcl = np.hstack((pcl, semantic_labels))

        calib_file = f'{self.data_path}/calib/{frame_name}.txt'
        img_file = f'{self.data_path}/image_2/{frame_name}.png'

        return pcl, f'{self.data_path}/label_2/{frame_name}.txt', instance, calib_file, img_file

    def delete_item(self, idx):
        self.velodyne_list = np.delete(self.velodyne_list, idx)

    def save_data(self, point_cloud, added_points, folder, name, idx, additional_anno_lines):
        scene_pcl = self.remove_space_for_spherical(point_cloud)
        all_visible_parts = self.remove_space_for_spherical(added_points)

        ## CREATING ANNOTATION

        create_annotation(old_address=f'{self.data_path}/label_2/{name}.txt',
                          new_address=f'{self.save_output_folder}/{folder}/label_2/{name}.txt',
                          additional_annotations_lines=additional_anno_lines)

        ## SAVING PCL TO .BIN
        save_file = open(f'{self.save_output_folder}/{folder}/velodyne/{name}.bin', 'wb')
        save_file.write(scene_pcl.astype(np.float32))
        save_file.close()

        check_file = open(f'{self.save_output_folder}/{folder}/check/{name}.bin', 'wb')
        check_file.write(all_visible_parts.astype(np.float32))
        check_file.close()

        self.delete_item(idx)

    def remove_space_for_spherical(self, point_cloud):
        '''
        :param point_cloud: point-cloud as 2D numpy array with space for spherical coordination
        (format of add_space_for_spherical output)
        :return: point-cloud as 2D numpy array (N x 4). Where N is number of points.
        Columns are ordered x, y, z. intensity
        '''
        points_num = len(point_cloud)
        labels = np.zeros((points_num, 1))
        pcl = np.ones((points_num, 4)) * -1
        pcl[:, 0:3] = point_cloud[:, 0:3]  # X, Y, Z
        pcl[:, 3] = point_cloud[:, 6]  # Intensity
        return pcl

    def create_velodyne_list(self):

        train_txt = open(self.train_txt_path)
        velodyne_address = []

        quit = False
        while not quit:
            frame_name = train_txt.readline()

            if len(frame_name) == 0:
                train_txt.close()
                quit = True

            else:
                frame_name = int(frame_name)
                velodyne_address.append(f'{self.data_path}/velodyne/{frame_name:06d}.bin')

        self.velodyne_list = velodyne_address

    def create_directories(self, save_folder):
        folder_number = 0

        save_output_folder = self.config['path']['output_path']

        if not os.path.exists(f'{save_output_folder}/{save_folder}'):
            os.mkdir(f'{save_output_folder}/{save_folder}')

        if os.path.exists(f'{save_output_folder}/{save_folder}/{folder_number:02d}'):
            print(f'Default save folder is already existing do you want change name? [yes/no]')
            quit = False
            while not quit:
                choice = input()
                if choice == 'yes':
                    quit1 = False
                    while not quit1:
                        print(f'Write name of the save folder.')
                        folder_number = input()
                        try:
                            folder_number = int(folder_number)

                            if not 0 <= folder_number <= 99:
                                print(f'Input must be number between 0 and 99')
                                continue
                        except:
                            print(f'Input must be number between 0 and 99')
                            continue
                        print(f'Is this what you want: \'{folder_number:02d}\'? [yes/no]')
                        choice = input()

                        if choice == 'yes':
                            quit = True
                            quit1 = True
                            if not os.path.exists(f'{save_output_folder}/{save_folder}/{folder_number:02d}'):
                                os.mkdir(f'{save_output_folder}/{save_folder}/{folder_number:02d}')

                elif choice == 'no':
                    quit = True

                else:
                    print(f'Wrong input. Try again.')

        else:
            os.mkdir(f'{save_output_folder}/{save_folder}/{folder_number:02d}')

        save_folder = f'{save_folder}/{folder_number:02d}'

        if not os.path.exists(f'{save_output_folder}/{save_folder}/velodyne'):
            os.mkdir(f'{save_output_folder}/{save_folder}/velodyne')

        if not os.path.exists(f'{save_output_folder}/{save_folder}/check'):
            os.mkdir(f'{save_output_folder}/{save_folder}/check')

        if not os.path.exists(f'{save_output_folder}/{save_folder}/label_2'):
            os.mkdir(f'{save_output_folder}/{save_folder}/label_2')
        if not os.path.exists(f'{save_output_folder}/{save_folder}/added_objects'):
            os.mkdir(f'{save_output_folder}/{save_folder}/added_objects')

        create_read_me(f'{save_output_folder}/{save_folder}', self.config)

        return f'{save_folder}', folder_number