import numpy as np


class LidarToImageProjector:
    def __init__(self, image: np.ndarray, normalize_color_to_one=False) -> None:
        self.image = image

        self.image_width = image.shape[1]
        self.image_height = image.shape[0]

    def project_lidar_to_image(self, points: np.ndarray, calibration_ext: np.ndarray,
                               calibration_int: np.ndarray) -> np.ndarray:
        assert points.shape[1] == 3

        assert calibration_ext.shape == (4, 4)
        assert calibration_int.shape == (3, 3)

        points_in_homogenous_coordinates = np.vstack([points.T, np.ones((1, points.shape[0],))])
        points_in_camera_coordinates = (calibration_ext[0:3] @ points_in_homogenous_coordinates).T

        points_on_image_plane = (calibration_int @ calibration_ext[0:3, :] @ points_in_homogenous_coordinates).T
        depths = points_in_camera_coordinates[:, 2]

        points_on_image = np.vstack(
            (points_on_image_plane[:, 0] / points_on_image_plane[:, 2],
             points_on_image_plane[:, 1] / points_on_image_plane[:, 2])).transpose()

        points_on_image_indexes = self.retrieve_only_valid_indexes(points_on_image=points_on_image,
                                                                   depths=depths)

        points_on_image_in_px = np.round(points_on_image[points_on_image_indexes])
        depths = depths[points_on_image_indexes]

        return points_on_image_in_px, depths

    def retrieve_only_valid_indexes(self, points_on_image: np.ndarray, depths: np.ndarray) -> np.ndarray:
        return np.where(
            (points_on_image[:, 0] >= 0)
            & (points_on_image[:, 1] >= 0)
            & (points_on_image[:, 0] <= self.image_width)
            & (points_on_image[:, 1] <= self.image_height)
            & (depths > 0)
        )
