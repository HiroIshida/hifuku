import argparse
import pickle
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from hifuku.domain import DummyDomain
from hifuku.script_utils import load_library


def sample_colors(n_colors, cmap_name="hsv"):
    cmap = plt.cm.get_cmap(cmap_name)  # get the colormap
    colors = cmap(np.linspace(0, 1, n_colors))  # sample the colormap
    return colors


parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int)
parser.add_argument("-margin", type=str, default="default")
parser.add_argument("--active", action="store_true")
parser.add_argument("--legend", action="store_true")
args = parser.parse_args()
n_step: Optional[int] = args.n
margin_mode: Literal["default", "latest", "all"] = args.margin
show_active_sampling: bool = args.active

# domain = EightRooms_SQP_Domain
# domain = EightRooms_Lightning_Domain
domain = DummyDomain
config = domain.solver_config
world = domain.task_type.get_world_type().sample()

lib = load_library(domain, "cpu")

fig, ax = plt.subplots()
world.visualize((fig, ax))

sdf = world.get_exact_sdf()
sampled_colors = sample_colors(10)
print(len(lib.predictors))

if n_step > 0:
    margins_history = lib._margins_history
    margins_at_time = margins_history[n_step - 1]
    for i in range(n_step):
        pred, margin = lib.predictors[i], margins_at_time[i]

        sol = pred.initial_solution
        assert sol is not None
        arr = sol.numpy()

        n_grid = 600
        xlin = np.linspace(-2.2, 2.2, n_grid)
        ylin = np.linspace(-2.2, 2.2, n_grid)
        X, Y = np.meshgrid(xlin, ylin)
        pts = list(zip(X.flatten(), Y.flatten()))
        n_batch = len(pts)
        if domain.task_type.get_dof() != 2:
            for i in range(n_batch):
                pts[i] = np.hstack((np.array([0.0, 0.0]), pts[i]))
        pts = np.array(pts)
        pts_torch = torch.from_numpy(np.array(pts)).float()

        mesh_torch = torch.empty((n_batch, 0))
        iter_preds, _ = pred.forward((mesh_torch, pts_torch))

        iters_main = iter_preds.cpu().detach().numpy().flatten() - (config.n_max_call - margin)
        assert np.any(iters_main > 0)

        if domain.task_type.get_dof() == 2:
            values = -sdf(pts)
        else:
            values = -sdf(pts[:, 2:])
        iters_main = np.maximum(values, iters_main)
        iters_main = iters_main.reshape(n_grid, n_grid)

        c = sampled_colors[i]

        ax.contourf(X, Y, iters_main, levels=[-np.inf, 0], colors=[c], alpha=0.3)
        ax.scatter(arr[-1, 0], arr[-1, 1], color="black", label="guiding trajectory")

        CS_main = ax.contour(X, Y, iters_main, levels=[0], colors=["black"])
        CS_main.collections[0].set_label("translated boundary")

        if margin_mode == "default":
            continue
        if margin_mode == "latest" and i != n_step - 1:
            continue

        iters_raw = iter_preds.cpu().detach().numpy().flatten() - config.n_max_call
        if domain.task_type.get_dof() == 2:
            values = -sdf(pts)
        else:
            values = -sdf(pts[:, 2:])
        iters_raw = np.maximum(values, iters_raw)
        iters_raw = iters_raw.reshape(n_grid, n_grid)

        CS_raw = ax.contour(X, Y, iters_raw, levels=[0], colors=["red"])
        CS_raw.collections[0].set_label("raw")

if show_active_sampling:
    p = Path("/tmp/hifuku-debug-data/feasible_solutions-{}.pkl".format(n_step))
    if p.exists():
        with p.open(mode="rb") as f:
            solutions = pickle.load(f)
        pts = []
        for traj in solutions:
            pts.append(traj.numpy()[-1])
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], c="b", marker="x", s=5, label="candidates")
    if n_step < len(lib.predictors):
        p_selected = lib.predictors[n_step].initial_solution.numpy()[-1]
        ax.scatter(p_selected[0], p_selected[1], c="orange", label="selected")

ax.set_xlim(-1.0, 2.2)
ax.set_ylim(-1.8, 1.6)
if args.legend:
    legend = ax.legend()
    legend.get_frame().set_alpha(1.0)
    plt.savefig("./figs/caption-algo-expl-seq-{}.png".format(n_step), dpi=300)
else:
    plt.savefig("./figs/algo-expl-seq-{}.png".format(n_step), dpi=300)
# plt.show()
