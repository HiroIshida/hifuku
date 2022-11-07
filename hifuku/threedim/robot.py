import numpy as np
from skrobot.models.pr2 import PR2

from hifuku.threedim.camera import Camera


def get_pr2_kinect_camera() -> Camera:
    pr2 = PR2()
    pr2.reset_manip_pose()

    camera = Camera()
    pr2.head_plate_frame.assoc(camera, relative_coords="local")
    camera.translate(np.array([-0.2, 0.0, 0.17]))
    return camera
