import time

import numpy as np
import skrobot
from skplan.viewer.skrobot_viewer import set_robot_config
from skrobot.model import Axis

from hifuku.threedim.tabletop import (
    TabletopPlanningProblem,
    create_simple_tabletop_world,
)

np.random.seed(0)

world = create_simple_tabletop_world(with_obstacle=True)
pose = world.sample_standard_pose()
pose.translate([0.25, -0.15, 0.0])

problem = TabletopPlanningProblem(world, [pose])
result = problem.solve()[0]

pr2 = problem.setup_pr2()
efkin, colkin = problem.setup_kinmaps()

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(world.table)
for obs in world.obstacles:
    viewer.add(obs)
axis = Axis()
axis.newcoords(pose)
viewer.add(axis)
viewer.add(pr2)
viewer.show()

for av in result.traj_solution:
    print(av)
    set_robot_config(pr2, efkin.control_joint_names, av, with_base=True)
    viewer.redraw()
    time.sleep(1.0)

time.sleep(100)
