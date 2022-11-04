import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_angle


def skcoords_to_pose_vec(co: Coordinates) -> np.ndarray:
    pos = co.worldpos()
    rot = co.worldrot()
    ypr = rpy_angle(rot)[0]
    rpy = np.flip(ypr)
    return np.hstack((pos, rpy))
