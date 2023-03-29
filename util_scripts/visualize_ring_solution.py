import matplotlib.pyplot as plt
import numpy as np
import torch
from rpbench.ring import RingObstacleFreePlanningTask, RingObstacleFreeWorld

from hifuku.script_utils import load_library

task = RingObstacleFreePlanningTask.sample
world = RingObstacleFreeWorld.sample()

lib = load_library("ring_rrt", "cpu")

fig, ax = plt.subplots()
world.visualize((fig, ax))

sdf = world.get_exact_sdf()
cmap = plt.cm.get_cmap(plt.cm.viridis, 143)

for pred, margin in zip(lib.predictors, lib.margins):

    sol = pred.initial_solution
    assert sol is not None
    arr = sol.numpy()

    xlin = np.linspace(0, 1, 100)
    ylin = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(xlin, ylin)
    pts = np.array(list(zip(X.flatten(), Y.flatten())))
    n_batch = len(pts)
    for i in range(n_batch):
        pts[i] = np.hstack((np.array([0.5, 0.05]), pts[i]))
    pts_torch = torch.from_numpy(np.array(pts)).float()

    mesh_torch = torch.empty((n_batch, 0))
    iter_preds, _ = pred.forward((mesh_torch, pts_torch))
    iters = iter_preds.cpu().detach().numpy().flatten() - (100 - margin)

    values = -sdf(pts[:, 2:])
    iters = np.maximum(values, iters)
    iters = iters.reshape(100, 100)

    c = np.random.rand(3)
    ax.contourf(X, Y, iters, levels=[-np.inf, 0], colors=[c], alpha=0.4)

    ax.plot(arr[:, 0], arr[:, 1], color=c)
    ax.scatter(arr[-1, 0], arr[-1, 1], color="black")
plt.show()
