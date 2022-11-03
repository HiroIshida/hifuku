import time

import numpy as np
import skrobot
from skrobot.model.primitives import Axis
from skrobot.models.pr2 import PR2

from hifuku.camera import Camera, create_synthetic_esdf
from hifuku.tabletop import create_tabletop_world

world = create_tabletop_world()
union_sdf = world.get_union_sdf()

pr2 = PR2()
pr2.reset_manip_pose()

camera_frame = Axis()
camera_frame.newcoords(pr2.head_plate_frame.copy_worldcoords())
camera_frame.translate(np.array([-0.2, 0.0, 0.17]))
camera = Camera()
camera.newcoords(camera_frame.copy_worldcoords())

esdf = create_synthetic_esdf(union_sdf, camera)
grid = esdf.get_voxel_info().get_boundary_grid(0.04)
grid_sdf = esdf.get_grid_sdf(grid)
grid_sdf.render_volume(isomin=-0.1, isomax=0.2)


grid = world.get_grid()
grid_sdf = esdf.get_grid_sdf(grid)
grid_sdf.render_volume(isomin=-0.05, isomax=0.1)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(world.table)
viewer.add(pr2)
viewer.add(camera_frame)
for obs in world.obstacles:
    viewer.add(obs)
viewer.show()

time.sleep(10)
