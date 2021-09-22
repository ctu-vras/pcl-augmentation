import open3d as o3d
import os
import glob

from tools.closing import *
from tools.find_spot import *
from tools.cutout import *

TEST = False

NUMROW = 112
NUMCOLUMN = 360*4

RGB_CLASS = np.array([[255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 0, 255], [123, 123, 123], [0, 255, 0], [0, 0, 0]])

DATA_PATH = ''              # path to KITTI dataset

SAMPLE_PATH = ''            # path to cutout objects (same path as 'save_path' in object_cutout/kitti_cutout.py)

TRAIN_TXT_PATH = ''         # path to txt, which contains names of frames in training dataset

SAVE_OUTPUT_FOLDER = ''     # path to output directory


def create_image(labels, filename):
    """
    Function, which creates a Image of classes.
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


def visualization(pcl, colors=True):
    """
    Function, which visualizes point-cloud.
    :param pcl: numpy 2D array, with point-cloud
    :param colors: bool, if True points are colored based on classes, otherwise based on z coordinate
    """
    xyz = pcl[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if colors:
        rgb = np.ones((len(pcl), 3))
        for i in range(len(pcl)):
            rgb[i, :] = RGB_CLASS[int(pcl[i][7])]
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud("current_directory.ply", pcd)
    cloud = o3d.io.read_point_cloud("current_directory.ply")  # Read the point cloud
    vis = o3d.visualization.draw_geometries([cloud])  # Visualize the point cloud


def add_space_for_spherical(point_cloud):
    """
    Function, which adds space for spherical coordinates and index of point in FoV.
    :param point_cloud: numpy 2D array, point-cloud (N x 5). N is number of points.
    Columns are ordered x, y, z, intensity, semantic label.
    :return: numpy 2D array, point-cloud with space for spherical coordination, semantic label, and points index in FoV
    """
    points_num = len(point_cloud)
    out = np.ones((points_num, 9))*-6
    out[:, 0:3] = point_cloud[:, 0:3]    # x, y, z
    out[:, 6:8] = point_cloud[:, 3:5]    # intensity, label
    return out  # x, y, z, r, azimuth, elevation, intensity, label, index


def remove_space_for_spherical(point_cloud):
    """
    Function, which removes space for spherical coordinates, semantic label, and index of point in FoV.
    :param point_cloud: numpy 2D array, point-cloud with space for spherical coordination, semantic label, and index of point in FoV
    (format of add_space_for_spherical output)
    :return: numpy 2D array, point-cloud (N x 4). N is number of points.
    Columns are ordered x, y, z. intensity
    """
    points_num = len(point_cloud)
    out = np.ones((points_num, 4))*-6
    out[:, 0:3] = point_cloud[:, 0:3]   # X, Y, Z
    out[:, 3] = point_cloud[:, 6]       # Intensity
    return out


def fill_spherical(point_cloud):
    """
    Function, which computes spherical coordinates of each point in point-cloud.
    :param point_cloud: numpy 2D array, point-cloud (N x 9). N is number of points.
    Columns are ordered x, y, z, _, _, _, intensity, semantic label, _.
    :return: numpy 2D array, point-cloud with computed spherical coordination and maximum and minimum elevation angle in point-cloud
    Columns are ordered x, y, z, radius, azimuth angle, elevation angle, intensity, semantic label, _.
    """
    point_cloud[:, 3] = np.sqrt(point_cloud[:, 0] ** 2 + point_cloud[:, 1] ** 2 + point_cloud[:, 2] ** 2)  # r (0;infty)
    point_cloud[:, 4] = np.arctan2(point_cloud[:, 1], point_cloud[:, 0]) + np.pi  # azimuth (0;2pi)
    point_cloud[:, 5] = np.arccos(point_cloud[:, 2] / point_cloud[:, 3])  # elevation

    min_elevation_angle = np.min(point_cloud[:, 5])
    max_elevation_angle = np.max(point_cloud[:, 5])

    return point_cloud, max_elevation_angle, min_elevation_angle


def geometrical_front_view(point_cloud, num_row, num_column, max_elevation_angle, min_elevation_angle, sample = False):
    """
    Function, which creates FoV and add FoV index to each point in point-cloud.
    :param point_cloud: numpy 2D array, point-cloud (N x 9). N is number of points.
    Columns are ordered x, y, z, radius, azimuth angle, elevation angle, intensity, semantic label, _.
    :param num_row: int, number of rows in front view
    :param num_column: int, number of columns in front view column
    :param max_elevation_angle: float, maximal elevation angle in original point-cloud
    :param min_elevation_angle: float, minimal elevation angle in original point-cloud
    :return: front view, labels for front view and point-cloud as 2D numpy array (N x 9). N is number of points.
    Columns are ordered x, y, z, radius, azimuth angle, elevation angle, intensity, semantic label, point index.
    """
    point_num = len(point_cloud)
    d_elevation = (max_elevation_angle - min_elevation_angle) / num_row       # resolution in row
    d_azimuth = 2 * math.pi / num_column           # resolution in column
    label = np.ones((num_row, num_column)) * -1
    train = np.zeros((num_row, num_column, 3))  # distance, intensity, arrived?

    for i in range(point_num):
        point = point_cloud[i]

        point_row = int((point[5] - min_elevation_angle - 0.00001) / d_elevation)
        point_column = int((point[4]%(2 * math.pi)) / d_azimuth)

        if sample and not(num_row > point_row >= 0):
            continue

        assert num_row > point_row >= 0, f'Rows in FoV went something wrong. Min elevation:{min_elevation_angle}, Max elevation:{max_elevation_angle}, Point elevation:{point[5]}'

        assert num_column > point_column >= 0, f'Column in FoV went something wrong. Point azimuth:{point[4]}'

        if label[point_row][point_column] != -1:

            point_cloud[i][8] = point_row * NUMCOLUMN + point_column

            if train[point_row][point_column][0] > point[3]:
                label[point_row][point_column] = point[7]  # label

                train[point_row][point_column][0] = point[3]  # distance
                train[point_row][point_column][1] = point[6]  # intensity
                train[point_row][point_column][2] = 1  # arrived

        else:
            label[point_row][point_column] = point[7]     # label

            train[point_row][point_column][0] = point[3]     # distance
            train[point_row][point_column][1] = point[6]     # intensity
            train[point_row][point_column][2] = 1            # arrived

            point_cloud[i][8] = point_row * NUMCOLUMN + point_column

    return train, label, point_cloud


def extract_anno(anno_path):
    """
    Function, which transforms original annotation txt from KITTI dataset to array of dictionaries
    :param anno_path: string, where is the txt file stored
    :return: numpy 1D array, in each index is stored dictionaries about one object in point-cloud
    """
    output_list = np.array([])
    txt_file = open(anno_path, 'r')

    end_of_file = False
    while not end_of_file:
        annotation = txt_file.readline()

        if len(annotation) == 0:
            end_of_file = True
            continue

        annotation_dictionary = read_label_line(annotation)

        if len(output_list) == 0:
            output_list = np.array([annotation_dictionary])
        else:
            output_list = np.append(output_list, [annotation_dictionary])

    txt_file.close()
    return output_list


def create_annotation_line(original_string, new_annotation_dict, rotation):
    """
    Function, which creates annotation in same format as they are in KITTI txt files, for object, which was added.
    :param original_string: string, original annotation line, which described object at the beginning
    :param new_annotation_dict: dictionary, of annotation, which described current object position
    :param rotation: int, which described about how many degree is object rotated from original position
    :return: string, annotation for added object in KITTI format
    """
    rotation = np.deg2rad(rotation)

    center_x = new_annotation_dict['center']['x']
    center_y = new_annotation_dict['center']['y']
    center_z = new_annotation_dict['center']['z']

    original_string = original_string.item()
    original_items = original_string.split(' ')
    sample_class = new_annotation_dict['class']
    sample_truncated = original_items[1]
    sample_occluded = f'3'
    sample_bbox = f'{original_items[4]} {original_items[5]} {original_items[6]} {original_items[7]}'
    sample_dimensions = f'{original_items[8]} {original_items[9]} {original_items[10]}'
    sample_location = f'{(center_y*-1):.02f} {(center_z*-1)-0.08:.02f} {center_x-0.27:.02f}'

    sample_rotation_y = float(original_items[14]) - rotation

    if sample_rotation_y < -np.pi:
        sample_rotation_y += 2 * np.pi
    elif sample_rotation_y > np.pi:
        sample_rotation_y -= 2 * np.pi

    assert -np.pi <= sample_rotation_y <= np.pi, f'Error in range of sample_rotation_y in file: {file}. Sample_rotation_y = {sample_rotation_y}'

    sample_alpha = (np.arctan2((center_y*-1), center_x-0.27) * -1) + sample_rotation_y

    if sample_alpha < -np.pi:
        sample_alpha += 2 * np.pi
    elif sample_alpha > np.pi:
        sample_alpha -= 2 * np.pi

    assert -np.pi <= sample_alpha <= np.pi, f'Error in range of alpha in file: {file}. Alpha = {sample_alpha}'

    sample_alpha = f'{sample_alpha:.02f}'

    sample_rotation_y = f'{sample_rotation_y:.02f}'

    return f'{sample_class} {sample_truncated} {sample_occluded} {sample_alpha} {sample_bbox} {sample_dimensions} {sample_location} {sample_rotation_y}\n'


def create_annotation(old_address, new_address, additional_line):
    """
    Function, which creates annotation txt for augmented point-cloud
    :param old_address: string, path to original txt file
    :param new_address: string, path, where to store new txt file
    :param additional_line: string, which described added object
    """
    old_txt = open(old_address, 'r')
    new_txt = open(new_address, 'w')

    end_of_file = False
    while not end_of_file:
        annotation = old_txt.readline()

        if len(annotation) == 0:
            end_of_file = True
            new_txt.write(additional_line)
            continue

        new_txt.write(annotation)

    old_txt.close()
    new_txt.close()


def create_output_folders(path):
    """
    Function, which creates directories where are stored all possible position of an object
    :param path: string, path which specifies where to create directories
    """
    if not os.path.exists(f'{path}'):
        os.mkdir(f'{path}')

    if not os.path.exists(f'{path}/velodyne'):
        os.mkdir(f'{path}/velodyne')

    if not os.path.exists(f'{path}/label_2'):
        os.mkdir(f'{path}/label_2')

    if not os.path.exists(f'{path}/check'):
        os.mkdir(f'{path}/check')


if __name__ == '__main__':
    quit = False
    save_folder = None
    folder_number = 0
    choice = 0
    inserted_class = ''
    while not quit:
        print(f'Choose insertion:')
        print(f'     Car ---------> 1')
        print(f'     Bikes -------> 2')
        print(f'     Pedestrian --> 3')
        choice = input()
        try:
            choice = int(choice)
            if choice == 1:
                print(f'Car insertion')
                save_folder = 'active_car_insertion'
                sample_folder = '/kitti_cars'
                inserted_class = 'Car'
                quit = True
            elif choice == 2:
                print(f'Bikes insertion')
                save_folder = 'active_bikes_insertion'
                sample_folder = '/kitti_bikes'
                inserted_class = 'Bike'
                quit = True
            elif choice == 3:
                print(f'Pedestrian insertion')
                save_folder = 'active_pedestrian_insertion'
                sample_folder = '/kitti_pedestrians'
                inserted_class = 'Pedestrian'
                quit = True
            else:
                print(f'Wrong input try again.')
        except:
            print(f'Wrong input try again.')

    if not os.path.exists(f'{SAVE_OUTPUT_FOLDER}/{save_folder}'):
        os.mkdir(f'{SAVE_OUTPUT_FOLDER}/{save_folder}')

    if os.path.exists(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/{folder_number:02d}'):
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
                        if not os.path.exists(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/{folder_number:02d}'):
                            os.mkdir(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/{folder_number:02d}')

            elif choice == 'no':
                quit = True

            else:
                print(f'Wrong input. Try again.')

    else:
        os.mkdir(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/{folder_number:02d}')

    save_folder = f'{save_folder}/{folder_number:02d}'

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
            current_directory.append(f'{DATA_PATH}/velodyne/{frame_name:06d}.bin')

    current_directory.sort()

    quit = False

    while not quit:
        quit = True
        for file in current_directory:
            if file.endswith(f'.bin'):
                frame_name = file.split('/')[-1].split('.')[0]
                print()
                print(f'{frame_name}.bin')
                if os.path.exists(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/data/{frame_name}.bin'):
                    print(f'Already done')
                    continue

                if inserted_class == 'Pedestrian':
                    map_data = np.load(f'{DATA_PATH}/maps/pedestrian_area/npz/{frame_name}.npz', allow_pickle=True)
                else:
                    map_data = np.load(f'{DATA_PATH}/maps/road_maps/npz/{frame_name}.npz', allow_pickle=True)

                calib_file = f'{DATA_PATH}/calib/{frame_name}.txt'
                img_file = f'{DATA_PATH}/image_2/{frame_name}.png'

                #### SCENE POINT CLOUD
                # original point cloud
                scene_pcl = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
                semantic_labels = np.fromfile(f'{DATA_PATH}/semantic_labels/labels/{frame_name}.label', dtype=np.uint32).reshape(-1, 1)
                semantic_labels = semantic_labels & 0xFFFF
                scene_pcl = np.hstack((scene_pcl, semantic_labels))

                scene_annotation = extract_anno(f'{DATA_PATH}/label_2/{frame_name}.txt')

                ###

                for i in range(len(scene_pcl)):
                    if scene_pcl[i][4] != 40 and scene_pcl[i][4] != 44 and scene_pcl[i][4] != 48: # road, parking, sidewalk
                        scene_pcl[i][4] = 1   # background

                ### SCENE FIELD OF VIEW (smooth)
                scene_pcl = add_space_for_spherical(scene_pcl)

                scene_pcl, max_elevation, min_elevation = fill_spherical(scene_pcl)

                scene_train, scene_label, scene_pcl = geometrical_front_view(scene_pcl, NUMROW, NUMCOLUMN, max_elevation, min_elevation)

                scene_train, scene_label = smooth_out(scene_train, scene_label)
                ###
                if TEST:
                    tmp_label = copy.deepcopy(scene_label)
                    for r in range(NUMROW):
                        for c in range(NUMCOLUMN):
                            if tmp_label[r][c] == -1:
                                tmp_label[r][c] = 0
                            else:
                                tmp_label[r][c] = 1
                    create_image(tmp_label, 'output/smooth_scene.png')

                scene_pcl_backup = copy.deepcopy(scene_pcl)

                ### SAMPLE POINT CLOUD

                sample_directory_list = glob.glob(f'{SAMPLE_PATH}{sample_folder}/*.npz')

                match_find = False
                number_of_tries = 0
                while not match_find:
                    sample_file = sample_directory_list[np.random.randint(len(sample_directory_list), size=1)[0]]

                    if sample_file.endswith('.npz'):

                        number_of_tries += 1
                        ### LOADING SAMPLE POINT CLOUD
                        object_name = sample_file.split('/')[-1].split('.')[0]
                        print(f'{object_name}')
                        sample_data = np.load(sample_file, allow_pickle=True)

                        # if len(sample_data['pcl']) < 50:
                        #     continue

                        possible_sample_pcl, possible_sample_annotation, possible_rotation = find_possible_places(scene_pcl,
                                                                                    scene_annotation, sample_data, map_data)

                        number_of_possibilites = 0

                        for sample_index in range(len(possible_sample_pcl)):
                            sample_pcl = possible_sample_pcl[sample_index]
                            sample_annotation = possible_sample_annotation[sample_index]
                            sample_rotation = possible_rotation[sample_index]
                            scene_pcl = copy.deepcopy(scene_pcl_backup)

                            if len(cutout_frame(sample_pcl, img_file, calib_file)) < 40:
                                continue

                            if TEST:
                                tmp = copy.deepcopy(scene_pcl)
                                tmp[:, 7] = 0
                                sample_pcl[:, 4] = 1
                                tmp = np.append(tmp, add_space_for_spherical(sample_pcl), axis=0)
                                visualization(tmp)
                                tmp = cutout_frame(tmp, img_file, calib_file)
                                visualization(tmp)

                            sample_pcl = add_space_for_spherical(sample_pcl)

                            sample_pcl, _, _ = fill_spherical(sample_pcl)

                            sample_train, sample_label, sample_pcl = geometrical_front_view(sample_pcl, NUMROW, NUMCOLUMN, max_elevation, min_elevation, sample=True)

                            # if TEST:
                            #     for row in range(NUMROW):
                            #         for column in range(NUMCOLUMN):
                            #             if sample_label[row][column] == 1 and sample_train[row][column][0] == 0:
                            #                 print('ROW:', row, 'COLUMN:', column, 'Sample distance is zero')

                            sample_train, sample_label = smooth_out(sample_train, sample_label)

                            if TEST:
                                tmp_label = copy.deepcopy(sample_label)
                                for r in range(NUMROW):
                                    for c in range(NUMCOLUMN):
                                        if tmp_label[r][c] == -1:
                                            tmp_label[r][c] = 0
                                create_image(tmp_label, 'output/smooth_sample.png')

                            first_part = True
                            visible_sample = np.array([])

                            for row in range(NUMROW):
                                for column in range(NUMCOLUMN):
                                    if sample_label[row][column] == 1:
                                        if scene_train[row][column][0] > sample_train[row][column][0] or scene_label[row][column] == -1:
                                            scene_pcl = scene_pcl[scene_pcl[:, 8] != row * NUMCOLUMN + column]      # all without covered pixel
                                            visible_part = sample_pcl[sample_pcl[:, 8] == row * NUMCOLUMN + column] # sample points in pixel
                                            if first_part:
                                                first_part = False
                                                visible_sample = visible_part
                                            else:
                                                visible_sample = np.append(visible_sample, visible_part, axis=0)

                            if TEST:
                                tmp = copy.deepcopy(scene_pcl)
                                tmp[:, 7] = 0
                                tmp = np.append(tmp, visible_sample, axis=0)
                                visualization(tmp)
                                tmp = cutout_frame(tmp, img_file, calib_file)
                                visualization(tmp)
                                tmp = copy.deepcopy(scene_pcl)
                                tmp[:, 7] = 1
                                _, tmp_label, _ = geometrical_front_view(tmp, NUMROW, NUMCOLUMN, max_elevation, min_elevation)
                                for r in range(NUMROW):
                                    for c in range(NUMCOLUMN):
                                        if tmp_label[r][c] == -1:
                                            tmp_label[r][c] = 0
                                create_image(tmp_label, 'output/scene_FoV.png')

                                tmp = copy.deepcopy(visible_sample)
                                tmp[:, 7] = 1
                                _, tmp_label, _ = geometrical_front_view(tmp, NUMROW, NUMCOLUMN, max_elevation,
                                                                         min_elevation)
                                for r in range(NUMROW):
                                    for c in range(NUMCOLUMN):
                                        if tmp_label[r][c] == -1:
                                            tmp_label[r][c] = 0
                                create_image(tmp_label, 'output/sample_FoV.png')

                            name_of_object = sample_annotation['class']
                            if len(visible_sample) == 0:
                                print(f'{RED}No visible part of {name_of_object}{DEFAULT}')
                            elif len(visible_sample) < 40:
                                print(f'It is visible just {YELLOW}{len(visible_sample)}{DEFAULT} points from {name_of_object}. It could be hard case. We try another one.')
                            else:
                                if number_of_possibilites == 0:
                                    create_output_folders(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/{frame_name}')

                                print(f'It is visible {GREEN}{len(visible_sample)}{DEFAULT} points from {name_of_object}. Total number of possible placements is {number_of_possibilites+1}.')
                                match_find = True
                                scene_pcl = np.append(scene_pcl, visible_sample, axis=0)

                                ### SAVING DATA

                                scene_pcl = remove_space_for_spherical(scene_pcl)
                                visible_sample = remove_space_for_spherical(visible_sample)

                                ## CREATING ANNOTATION

                                sample_annotation = create_annotation_line(sample_data['anno'], sample_annotation, sample_rotation)

                                create_annotation(old_address=f'{DATA_PATH}/label_2/{frame_name}.txt',
                                                  new_address=f'{SAVE_OUTPUT_FOLDER}/{save_folder}/{frame_name}/label_2/{number_of_possibilites:03d}.txt',
                                                  additional_line=sample_annotation)

                                ## SAVING PCL TO .BIN
                                save_file = open(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/{frame_name}/velodyne/{number_of_possibilites:03d}.bin', 'wb')
                                save_file.write(scene_pcl.astype(np.float32))
                                save_file.close()

                                check_file = open(f'{SAVE_OUTPUT_FOLDER}/{save_folder}/{frame_name}/check/{number_of_possibilites:03d}.bin', 'wb')
                                check_file.write(visible_sample.astype(np.float32))
                                check_file.close()

                                number_of_possibilites += 1

                    if number_of_tries >= 100:
                        quit = False
                        break