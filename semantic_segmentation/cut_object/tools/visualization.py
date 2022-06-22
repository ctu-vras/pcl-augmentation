import open3d as o3d
import numpy as np

RGB_CLASS = np.array([[255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 0, 255], [123, 123, 123], [0, 255, 0], [0, 0, 0]])


def visualization(pcl, colors=True, config=None):
    xyz = pcl[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if colors:
        rgb = np.ones((len(pcl), 3))
        for i in range(len(pcl)):
            rgb[i, :] = RGB_CLASS[int(pcl[i][4])]
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud("current_directory.ply", pcd)
    cloud = o3d.io.read_point_cloud("current_directory.ply")  # Read the point cloud
    vis = o3d.visualization.draw_geometries([cloud])  # Visualize the point cloud