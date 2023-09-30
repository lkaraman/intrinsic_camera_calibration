from typing import Optional

import numpy as np
import open3d as o3d

from structs import PointcloudRingInfo


class ScanlineExtractor:
    def __init__(self, points_in_xyz: np.ndarray, color_as_hot: bool=False) -> None:
        if np.ndim(points_in_xyz) != 2:
            raise ValueError

        if np.shape(points_in_xyz)[1] != 3:
            raise ValueError

        self.points_in_xyz = points_in_xyz
        self.color_as_hot = color_as_hot

        self.points_as_o3d_pointcloud = None
        self.scanlines: Optional[list[PointcloudRingInfo]] = None


    def points_as_o3d(self):
        if self.scanlines is None:
            raise ValueError

        xyz_complete = np.empty((0, 3))
        rgb_complete = np.empty((0, 3))

        for i in range(0, len(self.scanlines)):
            scanline = self.scanlines[i]

            numel = np.shape(scanline.xyz)[0]

            ccc = np.atleast_1d(scanline.color)
            c = np.tile(ccc, (numel, 1))

            xyz_complete = np.vstack((xyz_complete, scanline.xyz))
            rgb_complete = np.vstack((rgb_complete, c))

        pointSet2 = o3d.geometry.PointCloud()
        #
        pointSet2.points = o3d.utility.Vector3dVector(xyz_complete)
        pointSet2.colors = o3d.utility.Vector3dVector(rgb_complete)
        #
        o3d.visualization.draw_geometries([pointSet2])


    def visualize(self) -> None:
        if self.scanlines is None:
            raise ValueError





