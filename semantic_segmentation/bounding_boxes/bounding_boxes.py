from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pathlib import Path


class Ransac_line():
    def __init__(self, pts: np.array, th: float = 0.01, max_iter: int = 1000):
        assert pts.shape[1] == 2, f"pts should be in form (N, 2)"
        self.pts = self.cart_to_hom(pts)
        self.th = th
        self.max_iter = max_iter
        self.line_best = None

    def fit(self):
        inl_best = -np.infty

        for idx in range(self.max_iter):

            line = self._new_line()
            d = self._distance_from_line(line)

            inl = np.sum(d < self.th)
            if inl > inl_best:
                inl_best = inl
                self.line_best = line
                self.inl_mask = d < self.th

        return self.line_best

    def _distance_from_line(self, line):
        return np.abs(line @ self.pts.T) / np.sum(line[:2] ** 2)

    def _new_line(self):
        rnd_choice = np.random.choice(self.pts.shape[0], size=2, replace=False)
        AB = self.pts[rnd_choice]
        model = np.cross(AB[0], AB[1])
        return model

    def plot_result(self, line: np.array = None, viewbox_th: int = 2):
        if line == None: line = self.line
        a, b, c = line
        y = lambda a, b, c, x: (-c - a * x) / b
        pts_act = self.hom_to_cart(self.pts)

        plt.xlim([pts_act[:, 0].min() - viewbox_th, pts_act[:, 0].max() + viewbox_th])
        plt.ylim([pts_act[:, 1].min() - viewbox_th, pts_act[:, 1].max() + viewbox_th])
        plt.scatter(pts_act[:, 0], pts_act[:, 1], label="scan points")
        plt.plot([-10000, 10000], [y(a, b, c, x=-10000), y(a, b, c, x=10000)], label='fitted line')
        plt.fill_between([-10000, 10000], [y(a, b, c - self.th, x=-10000), y(a, b, c - self.th, x=10000)],
                         [y(a, b, c + self.th, x=-10000), y(a, b, c + self.th, x=10000)], alpha=0.2, label="Threshold")
        plt.legend()
        plt.show()

    @staticmethod
    def cart_to_hom(points):
        """
        :param points: shape -> (N, 2)
        :return: shape -> (N, 3)
        """
        return np.hstack([points, np.ones((points.shape[0], 1))])

    @staticmethod
    def hom_to_cart(points):
        """
        :param points: shape -> (N, 3)
        :return: shape -> (N, 2)
        """
        points /= points[:, 2].reshape((-1, 1))
        return points[:, :2]


class BoundingBox():
    def __init__(self, car_label):
        self.pts = np.zeros((0, 3))
        self.bbox = {'label': None,
                     'area': None,
                     'length': None,
                     'width': None,
                     'height': None,
                     'center': None,
                     'unit_vector': None,
                     'unit_vector_angle': None
                     }
        self.car_label = car_label

    def create_bounding_box(self, pts, label: int = None, z: str = "center", refiment: bool = False):
        """
        :param pts:
        :param label:
        :param z: "center" / "bottom"
        :param refiment:
        :return:
        """

        if pts is not None: self.pts = pts

        if self.pts.shape[0] < 6: return False  # raise ValueError("Number of points is below 6.")

        z_min = min(self.pts[:, 2])
        z_max = max(self.pts[:, 2])
        if z == "center":
            z = (z_max + z_min) / 2
        elif z == "bottom":
            z = z_min
        else:
            raise AttributeError("z, should be 'bottom' or 'center'")
        height = z_max - z_min
        self.bbox['height'] = height
        self.bbox['label'] = label

        xy = self.pts[:, :2]
        convex_hull = [xy[idx] for idx in ConvexHull(xy).vertices]
        convex_hull.append(convex_hull[0])

        min_rectangle = {"area": np.infty}
        for idx in range(0, len(convex_hull) - 1):
            rectangle = self._bounding_area(convex_hull, idx=idx)
            if rectangle['area'] < min_rectangle['area']:
                min_rectangle = rectangle

        min_rectangle['height'] = height
        min_rectangle['unit_vector_angle'] = np.arctan2(min_rectangle['unit_vector'][1],
                                                        min_rectangle['unit_vector'][0])
        min_rectangle['center'] = self._to_xy_coordinates(min_rectangle['unit_vector_angle'],
                                                          min_rectangle['center'], z)
        min_rectangle['label'] = label

        if refiment and label == self.car_label: # label 1 for Waymo || label 10 for semantic kitti
            min_rectangle['length'] = np.max([3.35, min_rectangle['length']])
            min_rectangle['height'] = np.max([1.39, min_rectangle['height']])
            min_rectangle['width'] = np.max([1.51, min_rectangle['width']])

        self.bbox = min_rectangle
        return min_rectangle

    def _bounding_area(self, convex_hull, idx):
        vector_par = self.unit_vector(convex_hull[idx], convex_hull[idx + 1])
        vector_orth = self.orthogonal_vector(vector_par)

        dis_par = [np.dot(vector_par, pt) for pt in convex_hull]  # distances on parallel line
        dis_orth = [np.dot(vector_orth, pt) for pt in convex_hull]  # distances on orthogonal line

        min_p = min(dis_par)
        min_o = min(dis_orth)
        len_p = max(dis_par) - min_p
        len_o = max(dis_orth) - min_o

        return {'area': len_p * len_o,
                'length': len_p,
                'width': len_o,
                'center': (min_p + len_p / 2, min_o + len_o / 2),
                'unit_vector': vector_par,
                }

    def _to_xy_coordinates(self, unit_vector_angle, point, z):
        angle_orthogonal = unit_vector_angle + np.pi / 2
        x = point[0] * np.cos(unit_vector_angle) + point[1] * np.cos(angle_orthogonal)
        y = point[0] * np.sin(unit_vector_angle) + point[1] * np.sin(angle_orthogonal)
        return np.array([x, y, z])

    @staticmethod
    def orthogonal_vector(vector):
        return -1 * vector[1], vector[0]

    @staticmethod
    def unit_vector(pt_0, pt_1):
        vector = pt_1 - pt_0
        vector = vector / np.linalg.norm(vector)
        return vector

    def plot2D(self):
        if self.bbox is None: return None
        plt.clf()
        plt.axis("equal")
        self._plot_bbox2D()
        plt.scatter(self.pts[:, 0], self.pts[:, 1], s=1, c="b")
        plt.show()

    def _plot_bbox2D(self, save=False, filename=None):
        """
                        P2________P4
        unit vector --> |   bbox  |
                        P1_______P2

        """
        vector = self.bbox["unit_vector"]
        vector_perpendicular = np.array([-vector[1], vector[0]])
        length = self.bbox["length"]
        width = self.bbox["width"]
        center = self.bbox["center"][:2]

        p1 = center - vector * length / 2 - vector_perpendicular * width / 2
        p2 = p1 + vector * length
        p3 = p1 + vector_perpendicular * width
        p4 = p2 + vector_perpendicular * width

        points = np.vstack((p1, p2, p4, p3, p1))
        plt.plot(points[:, 0], points[:, 1], color="red", markersize=0.1)
        if save:
            plt.axis('equal')
            plt.savefig(filename)
            plt.clf()

    def plot3D(self):
        if self.bbox is None: return None

        # points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pts)

        # bounding box
        extent = np.array([self.bbox["width"], self.bbox["length"], self.bbox["height"]])
        print(self.bbox["unit_vector_angle"])
        r = R.from_euler('z', self.bbox["unit_vector_angle"] - np.pi / 2, degrees=False).as_matrix()
        ori_bbox = o3d.geometry.OrientedBoundingBox(center=self.bbox["center"], R=r, extent=extent)
        ori_bbox.color = (1., 0., 0.)

        # coord
        frame_pcl = o3d.geometry.TriangleMesh.create_coordinate_frame(size=np.max(np.abs(self.pts)) / 10)
        trans = tuple(self.bbox["center"].reshape(-1, 1))
        frame_pcl = frame_pcl.translate(trans, relative=False)

        o3d.visualization.draw_geometries([pcd, ori_bbox, frame_pcl])

    def save_txt(self, path):

        os.makedirs(Path(path).parent, exist_ok=True)

        f = open(path, "a")
        f.write(f"{self.bbox['label']:.0f} ")
        f.writelines("%f " % loc for loc in self.bbox['center'])
        dimension = [self.bbox['height'], self.bbox['width'], self.bbox['length']]
        f.writelines("%f " % dim for dim in dimension)
        f.write(f"{self.bbox['unit_vector_angle']:.08f}" + "\n")
        f.close()


if __name__ == "__main__":
    pts = np.random.random((10, 3))
    bbox = BoundingBox()
    bbox.create_bounding_box(pts)
    # bbox.plot3D()
