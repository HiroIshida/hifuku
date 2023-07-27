import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from rpbench.articulated.jaxon.common import (
    InteractiveTaskVisualizer,
    StaticTaskVisualizer,
)
from skrobot.utils.urdf import mesh_simplify_factor

from hifuku.domain import select_domain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-domain", type=str, default="humanoid_trr_sqp", help="")
    parser.add_argument("-s", type=float, default=1.0)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="debug")
    args = parser.parse_args()

    domain_name: str = args.domain
    domain = select_domain(domain_name)
    result_base_path = Path("./result")
    result_path: Path = result_base_path / "result-{}".format(domain.get_domain_name())

    t = np.array(
        [
            [0.55978835, 0.2607793, -0.78653109, -3.03393855],
            [-0.82067329, 0.04320946, -0.56976161, -2.58098895],
            [-0.11459645, 0.96443097, 0.23820276, 1.98767049],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    with mesh_simplify_factor(args.s):
        with result_path.open(mode="rb") as f:
            tasks, results = pickle.load(f)

        indices = []
        for i, (task, result) in enumerate(zip(tasks, results)):
            res_proposed = result["proposed-guaranteed"]
            if res_proposed.traj is not None:
                indices.append(i)

        i = indices[args.i]
        traj = results[i]["proposed-guaranteed"].traj
        if args.debug:
            vis = InteractiveTaskVisualizer(tasks[i])
            vis.show()
            vis.visualize_trajectory(traj)
            time.sleep(1000)
        else:
            vis = StaticTaskVisualizer(tasks[i])
            vis.viewer.camera_transform = t
            parent = Path("./figs").expanduser()
            parent.mkdir(exist_ok=True)
            fig_path = parent / "traj-{}-{}.gif".format(domain_name, i)
            vis.save_trajectory_gif(traj.resample(10), path=fig_path)
