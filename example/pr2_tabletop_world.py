import time

import numpy as np
import skrobot
from skplan.viewer.skrobot_viewer import set_robot_config

from hifuku.threedim.robot import setup_kinmaps, setup_pr2
from hifuku.threedim.tabletop import VoxbloxTabletopPlanningProblem

np.random.seed(8)

pr2 = setup_pr2()
efkin, colkin = setup_kinmaps()

# create grid sdf using voxbloxpy
problem = VoxbloxTabletopPlanningProblem.sample(1)
res = problem.solve()[0]
assert res.success

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
problem.add_elements_to_viewer(viewer)
viewer.add(pr2)
viewer.show()

problem.grid_sdf.render_volume
problem.grid_sdf.render_volume(isomin=-0.1, isomax=0.1)

for q in res.traj_solution:
    set_robot_config(pr2, efkin.control_joint_names, q, with_base=True)
    viewer.redraw()
    time.sleep(0.2)
time.sleep(30)
