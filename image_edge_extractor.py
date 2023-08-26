import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageEdgeExtractor:
    def __init__(self, image_array: np.ndarray) -> None:
        self.image_array = image_array

    def extracts_edges(self, visualize=False) -> np.ndarray:


        grayscale = cv2.cvtColor(self.image_array, cv2.COLOR_BGR2GRAY)
        s = cv2.Sobel(grayscale, ddepth=-1, dx=1, dy=0)
        pixel_indexes_edges = np.where(s >= 100)

        if visualize:
            image = self.image_array[:, :, ::-1]
            image[pixel_indexes_edges] = (255, 0, 0)
            plt.imshow(image)
            plt.show()

        return pixel_indexes_edges




if __name__ == '__main__':
    image = cv2.imread('/home/luka/PycharmProjects/standard-3d-importer/standard_3d_importer/visualization/data/zf/222_front.jpg')

    ie = ImageEdgeExtractor(image_array=image)
    res = ie.extracts_edges(visualize=True)
