import time

from skrobot.model import Axis
from skrobot.viewers import TrimeshSceneViewer

from hifuku.threedim.tabletop import TableTopWorld

world = TableTopWorld.sample(standard=True)

viewer = TrimeshSceneViewer(resolution=(640, 480))
viewer.add(world.table)

for _ in range(20):
    pose = world.sample_pose(standard=True)
    viewer.add(Axis.from_coords(pose))
for obs in world.obstacles:
    viewer.add(obs)
viewer.show()
print("==> Press [q] to close window")
while not viewer.has_exit:
    time.sleep(0.1)
    viewer.redraw()
