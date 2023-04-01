import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mohou.trainer import TrainCache
from quotdim.intrinsic_dimension import determine_average_id
from quotdim.neural import IsometricMap

with Path("./dataset.pkl").open(mode="rb") as f:
    raw_dataset = pickle.load(f)

X = []
for x, y, _ in raw_dataset:
    X.append(x)
    X.append(y)

pp = Path("./isometric_embedding/out2dim")
tcache = TrainCache.load(pp, IsometricMap)
model = tcache.best_model
model.put_on_device(torch.device("cpu"))
Z = model.forward(torch.from_numpy(np.array(X)).float()).detach().numpy()

radius_list = [10, 20, 40, 80]
dims = []
for radius in radius_list:
    dim = determine_average_id(Z, radius=radius, n_neighbour=200)
    dims.append(dim)
plt.plot(radius_list, dims)
plt.show()
