import numpy as np
import glob
import os


def create_read_me(save_folder, config):
    txt = open(f'{save_folder}/setting.txt', 'w')
    txt.write(f'Inserted classes:\n')

    if config['insertion']['random']:
        for c in config['insertion']['classes']:
            txt.write('     ' + config['labels'][c] + '\n')
        txt.write('Randomly inserted '+ str(config['insertion']['number_of_object']) + ' objects\n')
    else:
        for c in range(len(config['insertion']['classes'])):
            txt.write('     '+ str(config['insertion']['number_of_classes'][c])+'x   ' + config['labels'][config['insertion']['classes'][c]] + '\n')
    txt.close()


class SemanticKITTI():
    def __init__(self, config, sequence):
        self.velo_2_cam = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                               [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                               [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
                               [0, 0, 0, 1]])
        self.my_calib = np.array([[0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1]])

        self.sequence = sequence
        self.config = config
        self.data_path = config['path']['dataset_path']
        self.anno_path = config['path']['annotation_path']

        self.poses = np.loadtxt(f'{self.data_path}/sequences/{self.sequence}/poses.txt')

        self.velodyne_list = np.array([])

        self.create_velodyne_list()

    def __len__(self):
        return len(self.velodyne_list)

    def __getitem__(self, idx):

        file = self.velodyne_list[idx]

        frame_name = file.split('/')[-1].split('.')[0]

        pcl = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(f'{self.data_path}/sequences/{self.sequence}/labels/{frame_name}.label',
                                      dtype=np.uint32).reshape(-1, 1)
        instance = labels >> 16
        semantic_labels = labels & 0xFFFF
        pcl = np.hstack((pcl, semantic_labels))

        transform_matrix = self.create_transform_matrix(poses=self.poses, frame_number=int(frame_name))

        return pcl, transform_matrix, f'{self.anno_path}/sequences/{self.sequence}/bbox/{frame_name}.txt', instance

    def delete_item(self, idx):
        self.velodyne_list = np.delete(self.velodyne_list, idx)

    def create_transform_matrix(self, poses, frame_number):
        pose = poses[frame_number].reshape(3, 4)
        pose = np.vstack((pose, np.array([0, 0, 0, 1])))

        transform_matrix = np.dot(pose, self.velo_2_cam)
        return np.dot(np.linalg.inv(self.my_calib), transform_matrix)

    def save_data(self, point_cloud, added_points, folder, name, idx):
        point_cloud, pcl_labels = self.remove_space_for_spherical(point_cloud)
        added_points, added_labels = self.remove_space_for_spherical(added_points)
        added_points = np.hstack((added_points, added_labels))

        save_output_folder = self.config['path']['output_path']

        ## SAVING PCL TO .BIN
        save_file = open(f'{save_output_folder}/{folder}/velodyne/{name}.bin', 'wb')
        save_file.write(point_cloud.astype(np.float32))
        save_file.close()
        save_labels = open(f'{save_output_folder}/{folder}/labels/{name}.label', 'wb')
        save_labels.write(pcl_labels.astype(np.uint32))
        save_labels.close()

        check_file = open(f'{save_output_folder}/{folder}/check/{name}.bin', 'wb')
        check_file.write(added_points.astype(np.float32))
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
        labels[:, 0] = point_cloud[:, 7]
        return pcl, labels

    def create_velodyne_list(self):
        skip_scenes = 0
        order = False

        quit = False

        while not quit:
            print(
                f'Do you want to reverse order of making scenes?[yes/no] or skip scene?[0;7000] (1/4 = 1842; 1/2 = 3699)')
            tmp = input()
            if tmp == 'no':
                quit = True
                order = False
            elif tmp == 'yes':
                quit = True
                order = True
            else:
                try:
                    tmp = int(tmp)
                    skip_scenes = tmp
                    print(f'Skiping scenes lower then {skip_scenes}')
                    quit = True
                except:
                    print(f'Wrong input try again.')

        velodyne_address = np.array(glob.glob(f'{self.data_path}/sequences/{self.sequence}/velodyne/*.bin'))

        velodyne_address.sort()

        velodyne_address = velodyne_address[skip_scenes:]

        if order:
            velodyne_address = velodyne_address[::-1]

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

        if not os.path.exists(f'{save_output_folder}/{save_folder}/sequences'):
            os.mkdir(f'{save_output_folder}/{save_folder}/sequences')

        save_folder = f'{save_folder}/sequences'

        create_read_me(f'{save_output_folder}/{save_folder}', self.config)

        for s in self.config['split']['train']:

            if not os.path.exists(f'{save_output_folder}/{save_folder}/{s:02d}'):
                os.mkdir(f'{save_output_folder}/{save_folder}/{s:02d}')

            if not os.path.exists(f'{save_output_folder}/{save_folder}/{s:02d}/velodyne'):
                os.mkdir(f'{save_output_folder}/{save_folder}/{s:02d}/velodyne')

            if not os.path.exists(f'{save_output_folder}/{save_folder}/{s:02d}/check'):
                os.mkdir(f'{save_output_folder}/{save_folder}/{s:02d}/check')

            if not os.path.exists(f'{save_output_folder}/{save_folder}/{s:02d}/labels'):
                os.mkdir(f'{save_output_folder}/{save_folder}/{s:02d}/labels')

            if not os.path.exists(f'{save_output_folder}/{save_folder}/{s:02d}/added_objects'):
                os.mkdir(f'{save_output_folder}/{save_folder}/{s:02d}/added_objects')

        return f'{save_folder}', folder_number


class Waymo():
    def __init__(self, config):

        self.config = config
        self.data_path = config['path']['dataset_path']
        self.anno_path = config['path']['annotation_path']

        self.velodyne_list = np.array([])

        self.sequence = None

        self.save_subfolder = None

        self.sequence_names = []

        self.create_velodyne_list()

    def __len__(self):
        return len(self.velodyne_list)

    def __getitem__(self, idx):

        pcl_file = self.velodyne_list[idx]

        label_file = pcl_file.split('/')
        matrix_file = pcl_file.split('/')

        sequence = label_file[-3]
        frame_name = label_file[-1].split('.')[0]

        label_file[-2] = 'labels_v3_2'
        matrix_file[-2] = 'poses'

        label_file = '/'.join(label_file)
        matrix_file = '/'.join(matrix_file)

        pcl = np.load(pcl_file).reshape(-1, 6)
        pcl = pcl[:, :4]
        semantic_labels = np.load(label_file).reshape(-1, 2)
        instances = semantic_labels[:, 0].reshape(-1, 1)
        semantic_labels = semantic_labels[:, 1].reshape(-1, 1)

        assert len(pcl) == len(semantic_labels), f'ERROR in sequence: {sequence}. pcl: {pcl_file}, sem: {label_file}, len(pcl): {len(pcl)}, len(sem): {len(semantic_labels)}'

        pcl = np.hstack((pcl, semantic_labels))

        transform_matrix = np.load(matrix_file).reshape(4, 4)

        return pcl, transform_matrix, f'{self.anno_path}/{sequence}/bbox/{frame_name}.txt', instances, sequence

    def delete_item(self, idx):
        self.velodyne_list = np.delete(self.velodyne_list, idx)

        if len(self.velodyne_list) == 0:
            self.sequence = None

        else:
            self.sequence = self.velodyne_list[0].split('/')[-3]

    def save_data(self, point_cloud, added_points, folder, name, idx):
        point_cloud, pcl_labels = self.remove_space_for_spherical(point_cloud)
        added_points, added_labels = self.remove_space_for_spherical(added_points)
        added_points = np.hstack((added_points, added_labels))

        save_output_folder = self.config['path']['output_path']

        ## SAVING PCL TO .NPY
        np.save(f'{save_output_folder}/{folder}/lidar/{name}.npy', point_cloud.astype(np.float32))

        np.save(f'{save_output_folder}/{folder}/labels_v3_2/{name}.npy', pcl_labels.astype(np.uint32))

        np.save(f'{save_output_folder}/{folder}/check/{name}.npy', added_points.astype(np.float32))

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
        labels[:, 0] = point_cloud[:, 7]
        return pcl, labels

    def create_velodyne_list(self):

        sequence_list = glob.glob(f'{self.data_path}/*/', recursive=True)

        velodyne_address = []

        for idx, sequence in enumerate(sequence_list):

            if sequence.split('/')[-2] in self.config['split']['skip']:
                continue

            sequence_pcl_address = glob.glob(f'{sequence}lidar/*.npy')
            sequence_pcl_address.sort()
            velodyne_address = velodyne_address + sequence_pcl_address

            self.sequence_names.append(sequence.split('/')[-2])

        self.velodyne_list = velodyne_address

        self.sequence = self.sequence_names[0]

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

        self.save_subfolder = f'{save_folder}/{folder_number:02d}'

        create_read_me(f'{save_output_folder}/{save_folder}', self.config)

        for sequence in self.sequence_names:

            self.create_subdirectories(sequence)

        return f'{self.save_subfolder}', folder_number

    def create_subdirectories(self, sequence):

        save_output_folder = self.config['path']['output_path']

        if not os.path.exists(f'{save_output_folder}/{self.save_subfolder}/{sequence}'):
            os.mkdir(f'{save_output_folder}/{self.save_subfolder}/{sequence}')

        if not os.path.exists(f'{save_output_folder}/{self.save_subfolder}/{sequence}/lidar'):
            os.mkdir(f'{save_output_folder}/{self.save_subfolder}/{sequence}/lidar')

        if not os.path.exists(f'{save_output_folder}/{self.save_subfolder}/{sequence}/check'):
            os.mkdir(f'{save_output_folder}/{self.save_subfolder}/{sequence}/check')

        if not os.path.exists(f'{save_output_folder}/{self.save_subfolder}/{sequence}/labels_v3_2'):
            os.mkdir(f'{save_output_folder}/{self.save_subfolder}/{sequence}/labels_v3_2')

        if not os.path.exists(f'{save_output_folder}/{self.save_subfolder}/{sequence}/added_objects'):
            os.mkdir(f'{save_output_folder}/{self.save_subfolder}/{sequence}/added_objects')