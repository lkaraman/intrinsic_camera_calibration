import pickle

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import KMeans

from scanline_extractors.scanline_extractor import ScanlineExtractor
from structs import PointcloudRingInfo
from utils import convert_pointcloud_from_cartesian_to_spherical


class KmeansScanlineExtractor(ScanlineExtractor):
    """
    Groups lidar points to individual scanlines by using Kmeans on polar angles
    This is required since we are detecting pointcloud edges by scanning the lidar beams horizontally,
    and this information is not available in given pointcloud data
    """

    def extract_scanlines(self, n_clusters: int):
        if self.color_as_hot is True:
            norm = mpl.colors.Normalize(vmin=0, vmax=n_clusters)
            cmap = cm.hot
            self.m = cm.ScalarMappable(norm=norm, cmap=cmap)

        pointcloud_spherical = convert_pointcloud_from_cartesian_to_spherical(points=self.points_in_xyz)

        height_ang = 90 - np.rad2deg(pointcloud_spherical[:, 2])
        height_ang = np.atleast_2d(height_ang).T

        # km = (MiniBatchKMeans(n_clusters=n_clusters, n_init=1))
        km = (KMeans(n_clusters=n_clusters, n_init=1, random_state=0))
        cluster = km.fit_predict(height_ang)

        cluster_centers = km.cluster_centers_.flatten()

        self._extract_scanlines_from_clusters(cluster, cluster_centers, pointcloud_spherical)

    def _extract_scanlines_from_clusters(self, cluster, clusters_centers, pointcloud_spherical):
        scanlines: list[PointcloudRingInfo] = []

        indexes_of_clusters = clusters_centers.argsort()

        for i, j in enumerate(indexes_of_clusters):
            index_which_belong_to_same_line = np.where(cluster == j)
            pnts_clustered = self.points_in_xyz[index_which_belong_to_same_line]
            pnts_clustered_radial = pointcloud_spherical[index_which_belong_to_same_line]

            # Sort each scanline so the points are processed 'left to right'
            indexes = pnts_clustered_radial[:, 1].argsort()
            points_xyz = pnts_clustered[indexes]
            points_spherical = pnts_clustered_radial[indexes]

            if self.color_as_hot is True:
                colors_clustered = np.asarray(self.m.to_rgba(i))[0:3]
            else:
                colors_clustered = np.random.rand(3, )

            scanlines.append(
                PointcloudRingInfo(
                    xyz=points_xyz,
                    rtp=points_spherical,
                    color=colors_clustered
                )
            )

        self.scanlines = scanlines


if __name__ == '__main__':
    with open('/home/luka/Desktop/Customer/Panda/001/lidar/00.pkl', 'rb') as f:
        a = pickle.load(f)

    x = np.atleast_2d(a['x'].values).T
    y = np.atleast_2d(a['y'].values).T
    z = np.atleast_2d(a['z'].values).T

    xyz = np.hstack((x, y, z))

    kse = KmeansScanlineExtractor(points_in_xyz=xyz, color_as_hot=False)
    kse.extract_scanlines(n_clusters=64)
    kse.points_as_o3d()
