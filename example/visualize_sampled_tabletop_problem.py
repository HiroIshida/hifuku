import time

import skrobot
from skrobot.model import Axis

from hifuku.threedim.tabletop import TableTopWorld

world = TableTopWorld.sample()
target_pose = world.sample_pose()

axis = Axis()
axis.newcoords(target_pose)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(world.table)
viewer.add(axis)
for obs in world.obstacles:
    viewer.add(obs)

viewer.show()

time.sleep(100)
