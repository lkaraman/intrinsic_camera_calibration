import numpy as np

from structs import PointcloudRingInfo
from utils import azimuth_change, non_max_suppression, threshold_values, \
    calculate_convolution_fft, gaussian_derivative_eval


class PointcloudCornerExtractor:
    """
    Extracts horizontal edges from the pointcloud by iterating through each scanline individually
    Information about reflectance present in the pointcloud data is NOT used
    """

    def __init__(self, sigma_c=0.1, window_size=4, T=5, Taz=0.01) -> None:
        self.sigma_c = sigma_c
        self.window_size = window_size
        self.T = T
        self.Taz = Taz

    def find_corners(self, pointcloud_ring_infos: list[PointcloudRingInfo]) -> np.ndarray:
        """
        Extracts edges (pages 19, 20 from [1])
        :param pointcloud_ring_infos: grouped points by scanline
        :return: points which are detected as lidar corners
        """

        detected_corners = np.empty((0, 3))

        for ring_info in pointcloud_ring_infos:
            d = calculate_convolution_fft(distances=ring_info.rtp[:, 0], gaussian=gaussian_derivative_eval(sigma_c=self.sigma_c))
            d = non_max_suppression(input=d, window_size=self.window_size)
            _, indexes_from_distances = threshold_values(input=d, T=self.T)

            indexes_from_azimuth = azimuth_change(input=ring_info.rtp, Taz=self.Taz)

            indexes_of_detected_corners = np.union1d(indexes_from_distances, indexes_from_azimuth)

            pointcloud_corners = ring_info.xyz[indexes_of_detected_corners]

            if len(pointcloud_corners) != 0:
                detected_corners = np.vstack((detected_corners, pointcloud_corners))

        return detected_corners



