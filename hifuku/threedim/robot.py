from typing import Tuple

import numpy as np
from skplan.kinematics import (
    ArticulatedCollisionKinematicsMap,
    ArticulatedEndEffectorKinematicsMap,
)
from skplan.robot.pr2 import PR2Paramter
from skplan.space import Bounds
from skrobot.models.pr2 import PR2

from hifuku.threedim.camera import Camera

_cache = {"kinmap": None, "pr2": None, "bounds": None}


def setup_pr2() -> PR2:
    if _cache["pr2"] is None:
        pr2 = PR2(use_tight_joint_limit=False)
        pr2.reset_manip_pose()
        _cache["pr2"] = pr2
    return _cache["pr2"]  # type: ignore


def setup_bounds() -> Bounds:
    if _cache["bounds"] is None:
        bounds = PR2Paramter.rarm_default_bounds(with_base=True)
        _cache["bounds"] = bounds  # type: ignore
    return _cache["bounds"]  # type: ignore


def setup_kinmaps() -> Tuple[
    ArticulatedEndEffectorKinematicsMap, ArticulatedCollisionKinematicsMap
]:
    if _cache["kinmap"] is None:
        pr2 = setup_pr2()
        efkin = PR2Paramter.rarm_kinematics(with_base=True)
        efkin.reflect_skrobot_model(pr2)
        colkin = PR2Paramter.collision_kinematics(with_base=True)
        colkin.reflect_skrobot_model(pr2)
        _cache["kinmap"] = (efkin, colkin)  # type: ignore
    return _cache["kinmap"]  # type: ignore


def build_pr2_cache() -> None:
    setup_pr2()
    setup_bounds()
    setup_kinmaps()


def get_pr2_kinect_camera() -> Camera:
    pr2: PR2 = setup_pr2()
    camera = Camera()
    pr2.head_plate_frame.assoc(camera, relative_coords="local")
    camera.translate(np.array([-0.2, 0.0, 0.17]))
    return camera
