import time

import numpy as np
import torch
from mohou.file import get_project_path
from mohou.trainer import TrainCache
from mohou.utils import detect_device

from hifuku.nerual import IterationPredictor, VoxelAutoEncoder
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.types import ProblemInterface

pp = get_project_path("tabletop_ik")

device = detect_device()


ae_model: VoxelAutoEncoder = TrainCache.load(pp, VoxelAutoEncoder).best_model
pred: IterationPredictor = TrainCache.load_latest(pp, IterationPredictor).best_model

ae_model.put_on_device(device)
pred.put_on_device(device)


problem = TabletopPlanningProblem.sample(n_pose=20)
assert pred.initial_solution is not None
results = problem.solve(pred.initial_solution)
print([r.success for r in results])


def infer(problem: ProblemInterface):

    desc_np = np.array(problem.get_descriptions())
    desc = torch.from_numpy(desc_np).float()

    n_batch, _ = desc_np.shape

    mesh_np = problem.get_mesh()
    mesh = torch.from_numpy(np.expand_dims(mesh_np, axis=(0, 1))).float()

    ts = time.time()
    desc = desc.to(device)
    mesh = mesh.to(device)
    encoded: torch.Tensor = ae_model.encoder(mesh)
    encoded_repeated = encoded.repeat(n_batch, 1)
    print("time to encode mesh {}".format(time.time() - ts))

    ts = time.time()
    iterval, _ = pred.forward((encoded_repeated, desc))
    print(iterval.detach().cpu().numpy() < 80.0)
    print("time to inference {}".format(time.time() - ts))
    time.sleep(10)


infer(problem)
