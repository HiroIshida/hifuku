import time

import numpy as np
import torch
from mohou.file import get_project_path
from mohou.trainer import TrainCache
from mohou.utils import detect_device
from skplan.solver.optimization import IKConfig

from hifuku.nerual import IterationPredictor
from hifuku.threedim.tabletop import TabletopIKProblem

pp = get_project_path("tabletop_ik")

device = detect_device()

tcache = TrainCache.load(pp, IterationPredictor)
best_model = tcache.best_model
best_model.put_on_device(device)

while True:
    problem = TabletopIKProblem.sample()
    ik_config = IKConfig(disp=False)
    ret = problem.solve(np.zeros(10), config=ik_config)
    if ret.nit > 50:
        print(ret.nit)
        break

mesh = torch.from_numpy(problem.get_mesh()).float().to(device)
description = torch.from_numpy(problem.get_description()).float().to(device)
sample = (mesh.unsqueeze(dim=0), description.unsqueeze(dim=0))

ts = time.time()
out = best_model.forward(sample)
print(time.time() - ts)
print(out)
