import numpy as np
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

problem = TabletopIKProblem.sample(n_pose=10)
ik_config = IKConfig(disp=False)
ret = problem.solve(np.zeros(10), config=ik_config)
print([e.nit for e in ret])

out = best_model.infer(problem)
print(out)
