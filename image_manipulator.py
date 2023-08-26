import copy



import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.cm as cm

import matplotlib.pyplot as plt


from consts import CALIB_EXT, CALIB_INT, IMAGE_WIDTH, IMAGE_HEIGHT


class ImageManipulator:
    def __init__(self, image_array: np.ndarray):
        self.image_array = image_array

    def show_image_with_pointcloud_overlapped(self, bins, density: int, sublist: list[int]= [], fire: bool = False):
        image = copy.deepcopy(self.image_array)

        extrinsics = CALIB_EXT
        projection_matrix = CALIB_INT

        projection_matrix = np.reshape(projection_matrix, (3, 3))

        extrinsics = np.reshape(extrinsics, (4, 4))
        extrinsics_rub = extrinsics
        lidar_to_cam = np.linalg.inv(extrinsics_rub)

        projected_points = {}

        if sublist == []:
            sublist = range(1, len(bins)+1)

        for i in sublist:
            pc = bins[i][0]
            c = bins[i][2]

            old_points_homo_nx4 = np.hstack([pc, np.ones((pc.shape[0], 1))])
            points_in_cam_coordinates = (lidar_to_cam @ old_points_homo_nx4.transpose()).transpose()[:, 0:3]

            # Simplified
            points_homo = np.hstack([pc, np.ones((pc.shape[0], 1))])
            homog_points = (points_homo @ np.linalg.inv(extrinsics.T))[:, 0:3] @ projection_matrix.T

            points_on_image = np.vstack(
                (homog_points[:, 0] / homog_points[:, 2], homog_points[:, 1] / homog_points[:, 2])).transpose()

            points_on_image_indices = np.where(
                (points_on_image[:, 0] >= 0)
                & (points_on_image[:, 1] >= 0)
                & (points_on_image[:, 0] <= IMAGE_WIDTH)
                & (points_on_image[:, 1] <= IMAGE_HEIGHT)
                & (points_in_cam_coordinates[:, 2] > 0)
            )

            distance_for_plotting = homog_points[:, 2]
            distance_for_plotting = distance_for_plotting[points_on_image_indices]

            points_on_image = points_on_image[points_on_image_indices]
            points_on_image = np.round(points_on_image)

            projected_points[i] = (points_on_image, c)



        plt.figure()
        for i, (g, kkk) in enumerate(projected_points.items()):
            if fire == False:
                c = kkk[1] * 255
            else:
                norm = mpl.colors.Normalize(vmin=0, vmax=len(kkk[0]))
                cmap = cm.hot
                m = cm.ScalarMappable(norm=norm, cmap=cmap)
            # c = np.random.rand(3, ) * 255

            for k, i in enumerate(kkk[0]):
                if fire == True:
                    c = np.asarray(m.to_rgba(k))[0:3] * 255

                x, y = i
                x, y = int(x), int(y)

                white = np.tile(c, ((2 * density + 1) * (2 * density + 1)))
                white = np.reshape(white, (2 * density + 1, 2 * density + 1, 3))
                try:
                    image[y - density:y + density + 1, x - density:x + density + 1] = white
                except:
                    pass

            if len(kkk) == 0:
                continue
        plt.imshow(image)
        plt.show()

    def show_pointcloud_overlapped(self, points: np.ndarray, density):
        image = copy.deepcopy(self.image_array)

        extrinsics = CALIB_EXT
        projection_matrix = CALIB_INT

        projection_matrix = np.reshape(projection_matrix, (3, 3))

        extrinsics = np.reshape(extrinsics, (4, 4))
        extrinsics_rub = extrinsics
        lidar_to_cam = np.linalg.inv(extrinsics_rub)


        old_points_homo_nx4 = np.hstack([points, np.ones((points.shape[0], 1))])
        points_in_cam_coordinates = (lidar_to_cam @ old_points_homo_nx4.transpose()).transpose()[:, 0:3]

        # Simplified
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        homog_points = (points_homo @ np.linalg.inv(extrinsics.T))[:, 0:3] @ projection_matrix.T

        points_on_image = np.vstack(
            (homog_points[:, 0] / homog_points[:, 2], homog_points[:, 1] / homog_points[:, 2])).transpose()

        points_on_image_indices = np.where(
            (points_on_image[:, 0] >= 0)
            & (points_on_image[:, 1] >= 0)
            & (points_on_image[:, 0] <= IMAGE_WIDTH)
            & (points_on_image[:, 1] <= IMAGE_HEIGHT)
            & (points_in_cam_coordinates[:, 2] > 0)
        )

        distance_for_plotting = homog_points[:, 2]
        distance_for_plotting = distance_for_plotting[points_on_image_indices]

        points_on_image = points_on_image[points_on_image_indices]
        points_on_image = np.round(points_on_image)

        norm = mpl.colors.Normalize(vmin=0, vmax=len(points_on_image))
        cmap = cm.hot
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        plt.figure()
        for i, p in enumerate(points_on_image):
            c = np.asarray(m.to_rgba(i))[0:3] * 255
            # c = np.random.rand(3, ) * 255

            x, y = p
            x, y = int(x), int(y)

            white = np.tile(c, ((2 * density + 1) * (2 * density + 1)))
            white = np.reshape(white, (2 * density + 1, 2 * density + 1, 3))
            try:
                image[y - density:y + density + 1, x - density:x + density + 1] = white
            except:
                pass

        plt.imshow(image)
        plt.show()