from dataclasses import dataclass

import numpy as np

from utils import cartesian_to_polar

import matplotlib.pyplot as plt

import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.cm as cm

import matplotlib.pyplot as plt



@dataclass
class PointsRepresentation:
    points_xyz: np.ndarray
    points_spherical: np.ndarray


class PointcloudToBinManager:
    BINS = [-25, -19, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5.833, -5.666, -5.499, -5.332, -5.165, -4.998, -4.831,
            -4.664, -4.497, -4.33, -4.163, -3.996, -3.829, -3.662, -3.495,
            -3.328, -3.161, -2.994, -2.827, -2.66, -2.493, -2.326, -2.159,
            -1.992, -1.825, -1.658, -1.491, -1.324, -1.157, -0.99, -0.823,
            -0.656, -0.489, -0.322, -0.155, 0.012, 0.179, 0.346, 0.513,
            0.68, 0.847, 1.014, 1.181, 1.348, 1.515, 1.682, 1.849, 2, 3, 5, 8, 11, 15]

    def __init__(self, pointcloud_xyz: np.ndarray) -> None:
        if np.ndim(pointcloud_xyz) != 2:
            raise ValueError

        if np.shape(pointcloud_xyz)[1] != 3:
            raise ValueError

        self.pointcloud_xyz = pointcloud_xyz
        self.pointcloud_spherical = cartesian_to_polar(points=pointcloud_xyz)

        self.bins = self.to_bin()

    def to_bin(self):
        height_ang = 90 - np.rad2deg(self.pointcloud_spherical[:, 2])

        d = np.digitize(height_ang, self.BINS)

        bins = {}

        for i in range(1, len(self.BINS) + 1):
            points_xyz = self.pointcloud_xyz[np.where(d == i)]
            points_spherical = self.pointcloud_spherical[np.where(d == i)]

            # immediately sort them by azimuth angle
            indexes = points_spherical[:, 1].argsort()
            points_xyz = points_xyz[indexes]
            points_spherical = points_spherical[indexes]

            bins[i] = (points_xyz, points_spherical, np.random.rand(3, ))

        return bins

    def visualize(self, lines: list[int] = [], fire=False) -> None:
        if lines == []:
            lines = range(1, len(self.BINS) + 1)


        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i in lines:
            if fire is False:
                c=self.bins[i][2]
            else:
                ss = len(self.bins[i][0][:, 0])
                norm = mpl.colors.Normalize(vmin=0, vmax=ss)
                cmap = cm.hot
                m = cm.ScalarMappable(norm=norm, cmap=cmap)
                c = [np.asarray(m.to_rgba(i))[0:3] for i in range(0, ss)]
            xyz = self.bins[i][0]
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

            ax.scatter(x, y, z, c=c)
        plt.show(block=False)
