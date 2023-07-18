import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hifuku.domain import DummyDomain
from hifuku.script_utils import load_library

# domain = EightRooms_SQP_Domain
# domain = EightRooms_Lightning_Domain
domain = DummyDomain
config = domain.solver_config
world = domain.task_type.get_world_type().sample()

lib = load_library(domain, "cpu")

fig, ax = plt.subplots()
world.visualize((fig, ax))

sdf = world.get_exact_sdf()
cmap = plt.cm.get_cmap(plt.cm.viridis, 143)

n_step = 4

margins_at_time = lib._margins_history[n_step-1]
for i in range(n_step):
    pred, margin = lib.predictors[i], margins_at_time[i]

    sol = pred.initial_solution
    assert sol is not None
    arr = sol.numpy()

    n_grid = 400
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
    print(config.n_max_call)
    iters = iter_preds.cpu().detach().numpy().flatten() - (config.n_max_call - margin)

    if domain.task_type.get_dof() == 2:
        values = -sdf(pts)
    else:
        values = -sdf(pts[:, 2:])
    iters = np.maximum(values, iters)
    iters = iters.reshape(n_grid, n_grid)

    c = np.random.rand(3)
    ax.contourf(X, Y, iters, levels=[-np.inf, 0], colors=[c], alpha=0.4)
    # ax.contour(X, Y, iters, levels=[-margin], colors=["red"])
    ax.contour(X, Y, iters, levels=[0.1], colors=["black"])

    # ax.plot(arr[:, 0], arr[:, 1], color=c)
    ax.scatter(arr[-1, 0], arr[-1, 1], color="black")
    ax.set_xlim(-1.0, 2.2)
    ax.set_ylim(-1.8, 1.6)

with Path("/tmp/hifuku-debug-data/feasible_solutions-{}.pkl".format(n_step)).open(mode="rb") as f:
    solutions = pickle.load(f)

pts = []
for traj in solutions:
    pts.append(traj.numpy()[-1])
pts = np.array(pts)
ax.scatter(pts[:, 0], pts[:, 1])


plt.show()
