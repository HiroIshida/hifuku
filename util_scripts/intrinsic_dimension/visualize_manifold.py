import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from cloud_geodesic import FastMarchingTree
from mohou.trainer import TrainCache
from quotdim.neural import IsometricMap

with Path("./dataset.pkl").open(mode="rb") as f:
    raw_dataset = pickle.load(f)

X = []
for x, y, _ in raw_dataset:
    X.append(x)
    X.append(y)

pp = Path("./isometric_embedding/out3dim")
tcache = TrainCache.load(pp, IsometricMap)
model = tcache.best_model
model.put_on_device(torch.device("cpu"))
Z = model.forward(torch.from_numpy(np.array(X)).float()).detach().numpy()
Z = Z[:5000]
tree = FastMarchingTree(Z, 50)
tree.span_tree()
costs = tree.get_costs()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=1, c=costs)
plt.show()
