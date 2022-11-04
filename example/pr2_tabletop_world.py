import time

import skrobot

from hifuku.camera import RayMarchingConfig, create_synthetic_esdf
from hifuku.robot import get_pr2_kinect_camera
from hifuku.tabletop import create_tabletop_world

# create grid sdf using voxbloxpy
world = create_tabletop_world()
union_sdf = world.get_union_sdf()
camera = get_pr2_kinect_camera()

rm_config = RayMarchingConfig(max_dist=2.0)
esdf = create_synthetic_esdf(union_sdf, camera, rm_config=rm_config)

grid = world.get_grid()
esdf.get_grid_sdf(grid).render_volume(isomin=-0.1, isomax=0.1)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(world.table)
viewer.add(camera)
for obs in world.obstacles:
    viewer.add(obs)
viewer.show()

time.sleep(30)
