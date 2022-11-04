import time

import skrobot

from hifuku.camera import RayMarchingConfig, create_synthetic_esdf
from hifuku.robot import get_pr2_kinect_camera
from hifuku.tabletop import create_tabletop_world

world = create_tabletop_world()
union_sdf = world.get_union_sdf()
camera = get_pr2_kinect_camera()

rm_config = RayMarchingConfig(max_dist=2.0)
esdf = create_synthetic_esdf(union_sdf, camera, rm_config=rm_config)
info = esdf.get_voxel_info()
grid = info.get_boundary_grid(0.04)
grid_sdf = esdf.get_grid_sdf(grid)
grid_sdf.render_volume(isomin=-0.1, isomax=0.2)

info = info.filter(-0.02, 0.02)


grid = world.get_grid()
print(grid.lb)
print(grid.ub)

grid_sdf = esdf.get_grid_sdf(grid)
grid_sdf.render_volume(isomin=-0.1, isomax=0.1)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(world.table)
viewer.add(camera)
for obs in world.obstacles:
    viewer.add(obs)
viewer.show()

time.sleep(30)
