import time
from pathlib import Path

import skrobot
import torch
from skplan.viewer.skrobot_viewer import set_robot_config
from skrobot.model.primitives import LineString

from hifuku.library import SolutionLibrary
from hifuku.threedim.tabletop import TabletopPlanningProblem

# common setup
pr2 = TabletopPlanningProblem.setup_pr2()
efkin, colkin = TabletopPlanningProblem.setup_kinmaps()

# problem definition
# world = create_simple_tabletop_world(with_obstacle=True)
# pose = world.sample_standard_pose()
# pose.translate([0.0, -0.15, 0.0])
problem = TabletopPlanningProblem.sample(1)

# common viewer setup
viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
problem.add_elements_to_viewer(viewer)
viewer.add(pr2)


# solve
p = Path("~/.mohou/tabletop_solution_library").expanduser()
if p.exists():
    # should set device to cpu as putting on the voxel mesh on gpu is costly (takes 1sec)
    lib = SolutionLibrary.load(p, TabletopPlanningProblem, device=torch.device("cpu"))[0]
    linestrings = []
    for pred in lib.predictors:
        assert pred.initial_solution is not None
        avs = pred.initial_solution.reshape(-1, 10)
        endpoints, _ = efkin.map(avs)
        linestrings.append(LineString(endpoints[:, 0, :3]))
    for linestring in linestrings:
        viewer.add(linestring)

    ts = time.time()
    res = lib.infer(problem)[0]
    print("(using library) time to infer: {}".format(time.time() - ts))
    result = problem.solve(res.init_solution)[0]
    print(result.success)
    linestrings[res.idx].visual_mesh.colors = [[255, 0, 0, 255]]
    print("(using library) time to solve: {}".format(time.time() - ts))
else:
    ts = time.time()
    result = problem.solve()[0]
    print("(from scratch) time to solve: {}".format(time.time() - ts))

viewer.show()
for av in result.traj_solution:
    print(av)
    set_robot_config(pr2, efkin.control_joint_names, av, with_base=True)
    viewer.redraw()
    time.sleep(1.0)

time.sleep(100)
