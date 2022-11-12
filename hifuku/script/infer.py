import numpy as np
import torch
from mohou.file import get_project_path
from mohou.trainer import TrainCache
from mohou.utils import detect_device
from skplan.solver.optimization import IKConfig

from hifuku.nerual import IterationPredictor, VoxelAutoEncoder
from hifuku.threedim.tabletop import TabletopIKProblem
from hifuku.types import ProblemInterface

pp = get_project_path("tabletop_ik")

device = detect_device()


problem = TabletopIKProblem.sample(n_pose=10)
ik_config = IKConfig(disp=False)
ret = problem.solve(np.zeros(10), config=ik_config)
print([e.success for e in ret])

ae_model: VoxelAutoEncoder = TrainCache.load(pp, VoxelAutoEncoder).best_model
pred: IterationPredictor = TrainCache.load_latest(pp, IterationPredictor).best_model


def infer(problem: ProblemInterface):
    mesh = problem.get_mesh()
    mesh = torch.from_numpy(np.expand_dims(mesh, axis=(0, 1))).float()
    encoded = ae_model.encoder(mesh)

    desc = problem.get_descriptions()
    desc = torch.from_numpy(np.expand_dims(desc, axis=0)).float()
    iterval, _ = pred.forward((encoded, desc))
    print(iterval)

infer(problem)
