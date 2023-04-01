import matplotlib.pyplot as plt
import numpy as np
import torch

from hifuku.domain import EightRooms_SQP_Domain
from hifuku.script_utils import load_library

domain = EightRooms_SQP_Domain
# domain = EightRooms_Lightning_Domain
config = domain.solver_config
world = domain.task_type.get_world_type().sample()

lib = load_library(domain, "cpu")

fig, ax = plt.subplots()
world.visualize((fig, ax))

sdf = world.get_exact_sdf()
cmap = plt.cm.get_cmap(plt.cm.viridis, 143)

for pred, margin in zip(lib.predictors, lib.margins):

    sol = pred.initial_solution
    assert sol is not None
    arr = sol.numpy()

    xlin = np.linspace(-1, 1, 100)
    ylin = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(xlin, ylin)
    pts = list(zip(X.flatten(), Y.flatten()))
    n_batch = len(pts)
    for i in range(n_batch):
        pts[i] = np.hstack((np.array([0.0, 0.0]), pts[i]))
    pts = np.array(pts)
    pts_torch = torch.from_numpy(np.array(pts)).float()

    mesh_torch = torch.empty((n_batch, 0))
    iter_preds, _ = pred.forward((mesh_torch, pts_torch))
    print(config.n_max_call)
    iters = iter_preds.cpu().detach().numpy().flatten() - (config.n_max_call - margin)

    values = -sdf(pts[:, 2:])
    iters = np.maximum(values, iters)
    iters = iters.reshape(100, 100)

    c = np.random.rand(3)
    ax.contourf(X, Y, iters, levels=[-np.inf, 0], colors=[c], alpha=0.4)

    ax.plot(arr[:, 0], arr[:, 1], color=c)
    ax.scatter(arr[-1, 0], arr[-1, 1], color="black")
plt.show()
