import numpy as np

from image_edge_extractor import ImageEdgeExtractor
from kitti_frame_importer import KittiFrameImporter
from lidar_to_image_projector import LidarToImageProjector
from pointcloud_corner_extractor import PointcloudCornerExtractor
from scanline_extractors.kmeans_scanline_extractor import KmeansScanlineExtractor
from utils import Lf


class LossCalculator:

    def __init__(self, importer: KittiFrameImporter) -> None:
        self.importer = importer
        self.pointcloud = None
        self.edges_in_image = None
        self.corners_in_pointcloud = None
        self.lidar_to_image_projector = None

    def load_frame(self, frame_id: int) -> None:
        self.pointcloud, image = self.importer.load_pointcloud_and_image(frame_id=frame_id)

        ime = ImageEdgeExtractor(image_array=image, edge_detection_threshold=200)
        self.edges_in_image = ime.extracts_edges()
        image[self.edges_in_image] = (127, 0, 255)

        kse = KmeansScanlineExtractor(points_in_xyz=self.pointcloud, color_as_hot=True)
        kse.extract_scanlines(n_clusters=64)
        # kse.points_as_o3d()

        pce = PointcloudCornerExtractor(T=1, Taz=0.01)
        self.corners_in_pointcloud = pce.find_corners(pointcloud_ring_infos=kse.scanlines)

        self.lidar_to_image_projector = LidarToImageProjector(image=image)

    def compute_loss_function(self, opt_params):
        def loss(X):
            I, E = opt_params.parameter_to_intrinsics_and_extrinsics(X)

            p, _ = self.lidar_to_image_projector.project_lidar_to_image(points=self.corners_in_pointcloud,
                                                                        calibration_ext=E, calibration_int=I)

            lf = Lf(image_edges=np.asarray(self.edges_in_image)[::-1, :].T,
                    pointcloud_corners=p)
            return lf

        return loss
