import copy
import time
import uuid
from pathlib import Path

import numpy as np
import skrobot
import torch
from skplan.viewer.skrobot_viewer import set_robot_config
from skrobot.model.primitives import LineString

from hifuku.library import SolutionLibrary
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.threedim.viewer import SceneWrapper, set_robot_alpha


def save_image(viewer, postfix: str, uuidval: int):
    if not isinstance(viewer, SceneWrapper):
        return
    name = "image-{}-{}.png".format(uuidval, postfix)
    png = viewer.save_image(resolution=[640, 480], visible=True)
    p = Path(name)
    with p.open(mode="wb") as f:
        f.write(png)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize")
    args = parser.parse_args()
    visualize: bool = args.visualize

    # common setup
    pr2 = TabletopPlanningProblem.setup_pr2()
    efkin, colkin = TabletopPlanningProblem.setup_kinmaps()
    problem = TabletopPlanningProblem.sample(1)
    problem.grid_sdf
    TabletopPlanningProblem.cache_all()

    uuidval = str(uuid.uuid4())[-8:]

    # common viewer setup
    if visualize:
        viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    else:
        viewer = SceneWrapper()
        # transform = np.array(
        #    [[ 0.72054743,  0.25310975, -0.64555934, -1.14414428],
        #     [-0.68877715,  0.36865163, -0.62424515, -1.48286375],
        #     [ 0.07998397,  0.89444476,  0.4399672 ,  1.89720878],
        #     [ 0.        ,  0.        ,  0.        ,  1.        ]]
        #    )
        transform = np.array(
            [
                [0.83342449, 0.51809572, -0.19230299, -0.06748249],
                [-0.55223975, 0.76765106, -0.32518165, -0.77247049],
                [-0.02085362, 0.37721171, 0.92589225, 3.05147158],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        viewer.camera_transform = transform

    problem.add_elements_to_viewer(viewer)
    viewer.add(pr2)

    # solve
    p = Path("~/.mohou/tabletop_solution_library").expanduser()
    if p.exists():
        # should set device to cpu as putting on the voxel mesh on gpu is costly (takes 1sec)
        lib = SolutionLibrary.load(p, TabletopPlanningProblem, device=torch.device("cpu"))[0]
        lib.limit_thread = False
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
    save_image(viewer, "before", uuidval)
    set_robot_alpha(pr2, 100)

    if visualize:
        for av in result.traj_solution:
            print(av)
            set_robot_config(pr2, efkin.control_joint_names, av, with_base=True)
            viewer.redraw()
            time.sleep(1.0)
        time.sleep(100)
    else:
        for av in result.traj_solution:
            pr2 = copy.deepcopy(pr2)
            set_robot_alpha(pr2, 100)
            set_robot_config(pr2, efkin.control_joint_names, av, with_base=True)
            viewer.add(pr2)

        pr2 = copy.deepcopy(pr2)
        set_robot_alpha(pr2, 255)
        viewer.add(pr2)
        save_image(viewer, "after", uuidval)
