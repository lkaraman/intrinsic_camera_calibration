from dataclasses import dataclass

import numpy as np


@dataclass
class PointcloudRingInfo:
    xyz: np.ndarray
    rtp: np.ndarray
    color: tuple[float, float, float]
