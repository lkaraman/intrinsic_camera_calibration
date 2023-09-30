import numpy as np

from consts import EXTRINSICS


class OptimizationParameters:
    def __init__(self):
        self.X = np.asarray([
            [700],
            [700],
            [700],
            [100.0]
        ])

        # Ground truth
        # self.X = np.asarray([
        #     [7.215377e+02],
        #     [7.215377e+02],
        #     [6.095593e+02],
        #     [1.728540e+02]
        #
        # ])

    @property
    def number_of_parameters(self) -> int:
        return self.X.shape[0]

    def parameter_to_intrinsics_and_extrinsics(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        I = np.asarray([
            [X[0, 0], 0, X[2, 0]],
            [0, X[1, 0], X[3, 0]],
            [0, 0, 1]
        ])

        return I, EXTRINSICS
