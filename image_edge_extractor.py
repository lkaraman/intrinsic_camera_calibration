import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageEdgeExtractor:
    """
    Extracts edges (vertical only!) from an image using the Sobel operator
    """

    def __init__(self, image_array: np.ndarray, edge_detection_threshold: float = 50) -> None:
        self.image_array = image_array
        self.edge_detection_threshold = edge_detection_threshold

    def extracts_edges(self, visualize=False) -> np.ndarray:
        grayscale = cv2.cvtColor(self.image_array, cv2.COLOR_BGR2GRAY)
        sobel_edges = cv2.Sobel(grayscale, ddepth=-1, dx=1, dy=0, ksize=3)
        pixel_indexes_edges = np.where(sobel_edges >= self.edge_detection_threshold)

        if visualize:
            image = self.image_array[:, :, ::-1]
            image[pixel_indexes_edges] = (255, 0, 0)
            plt.imshow(image)
            plt.show()

        return pixel_indexes_edges


if __name__ == '__main__':
    image = cv2.imread(
        '/home/luka/PycharmProjects/standard-3d-importer/standard_3d_importer/visualization/data/zf/222_front.jpg')

    ie = ImageEdgeExtractor(image_array=image)
    res = ie.extracts_edges(visualize=True)
