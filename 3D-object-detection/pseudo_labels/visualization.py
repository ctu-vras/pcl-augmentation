import open3d as o3d
import open3d.ml as ml3d
import numpy as np
import yaml


RGB_CLASS = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [255, 0, 255], [123, 123, 123], [0, 255, 0], [0, 0, 255]])

def read_yaml(yaml_file):
    with open(yaml_file) as file:
        dict = yaml.load(file, Loader=yaml.FullLoader)
    return dict

def visualization_semantic(pcl, colors = True):
    """
    :param pcl: pointcloud in (x,y,z,class) format
    :param road: condition to color road
    """
    #rgb_dict = read_yaml("colors.yaml")
    xyz = pcl[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    rgb = np.ones((len(pcl), 3))
    if road:
        for i in range(len(pcl)):
            if pcl[i][4] != 40:
               pcl[i][4] = 0
            else:
                pcl[i][4] = 1
            rgb[i, :] = RGB_CLASS[int(pcl[i][4])]
            
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32) / 255.0)
    o3d.io.write_point_cloud("current_directory.ply", pcd)
    cloud = o3d.io.read_point_cloud("current_directory.ply")  # Read the point cloud
    vis = o3d.visualization.draw_geometries([cloud])  # Visualize the point cloud



def visualization_bbox(path_KITTI, frames):
    """
    :param path_KITTI: path to pseudo_labels KITTI
    :parma frames: number of frames to visualized
    """

    dataset = ml3d.datasets.KITTI(dataset_path=path_KITTI, use_cache=False)
    vis = ml3d.vis.Visualizer()
    vis.visualize_dataset(dataset, 'train', indices=range(frames))










