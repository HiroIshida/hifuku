import time

import numpy as np
import skrobot

from hifuku.threedim.tabletop import VoxbloxTabletopPlanningProblem

np.random.seed(4)
use_base = True

robot_model = skrobot.models.PR2()
robot_model.reset_manip_pose()

# create grid sdf using voxbloxpy
problem = VoxbloxTabletopPlanningProblem.sample(1)

visualize = True
if visualize:
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    problem.add_elements_to_viewer(viewer)
    viewer.add(robot_model)
    viewer.show()
    problem.grid_sdf.render_volume
    problem.grid_sdf.render_volume(isomin=-0.1, isomax=0.1)
    time.sleep(30)
