import time

import numpy as np
import torch
from mohou.file import get_project_path
from mohou.trainer import TrainCache
from mohou.utils import detect_device
from skplan.solver.inverse_kinematics import IKConfig

from hifuku.nerual import IterationPredictor, VoxelAutoEncoder
from hifuku.threedim.tabletop import TabletopIKProblem
from hifuku.types import ProblemInterface

pp = get_project_path("tabletop_ik")

device = detect_device()


problem = TabletopIKProblem.sample(n_pose=1000)
ik_config = IKConfig(disp=False)

ts = time.time()
ret = problem.solve_dummy(np.zeros(10), config=ik_config)
print("time to solve problem {}".format((time.time() - ts) / len(problem.target_pose_list)))

print([e.success for e in ret])

ae_model: VoxelAutoEncoder = TrainCache.load(pp, VoxelAutoEncoder).best_model
pred: IterationPredictor = TrainCache.load_latest(pp, IterationPredictor).best_model

ae_model.put_on_device(device)
pred.put_on_device(device)


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
    print("time to inference {}".format(time.time() - ts))
    time.sleep(10)


infer(problem)
