import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.models.pr2 import PR2

from hifuku.camera import Camera


def get_pr2_kinect_camera() -> Camera:
    pr2 = PR2()
    pr2.reset_manip_pose()

    camera_frame = Coordinates()
    camera_frame.newcoords(pr2.head_plate_frame.copy_worldcoords())
    camera_frame.translate(np.array([-0.2, 0.0, 0.17]))
    camera = Camera()
    camera.newcoords(camera_frame.copy_worldcoords())
    return camera
