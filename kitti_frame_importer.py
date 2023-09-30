import pathlib
import numpy as np
import open3d as o3d


class KittiFrameImporter:
    def __init__(self, path_to_root: pathlib) -> None:
        self.path_to_root = pathlib.Path(path_to_root)

    def _get_import_paths(self, frame_id: int):
        formatted_number = str(int(frame_id)).zfill(10)
        point_cloud_filename = self.path_to_root / f'velodyne_points/data/{formatted_number}.bin'
        image_filename = self.path_to_root / f'image_00/data/{formatted_number}.png'

        return str(point_cloud_filename), str(image_filename)

    @staticmethod
    def _load_image(image_filename):
        img = np.asarray(o3d.io.read_image(image_filename))
        img = np.stack((img,) * 3, axis=-1)

        return img

    @staticmethod
    def _load_pointcloud(point_cloud_filename):
        pc_data = np.fromfile(point_cloud_filename, '<f4')
        pc_data = np.reshape(pc_data, (-1, 4))

        pc_data = pc_data[:, :3]

        return pc_data

    def load_pointcloud_and_image(self, frame_id: int):
        pointcloud_filename, image_filename = self._get_import_paths(frame_id=frame_id)

        return self._load_pointcloud(pointcloud_filename), self._load_image(image_filename)
