import time

import torch
from mohou.file import get_project_path
from mohou.trainer import TrainCache

from hifuku.domain import TBRR_SQP_DomainProvider
from hifuku.library.core import SolutionLibrary
from hifuku.neuralnet import (
    AutoEncoderBase,
    IterationPredictor,
    NullAutoEncoder,
    VoxelAutoEncoder,
)

mesh_sampler_type = TBRR_SQP_DomainProvider.get_compat_mesh_sampler_type()
if mesh_sampler_type is None:
    ae_model: AutoEncoderBase = NullAutoEncoder()
else:
    ae_pp = get_project_path("hifuku-{}".format(mesh_sampler_type.__name__))
    ae_model = TrainCache.load(ae_pp, VoxelAutoEncoder).best_model

pp = get_project_path("TBRR_SQP")
pred: IterationPredictor = TrainCache.load_latest(pp, IterationPredictor).best_model

device = torch.device("cpu")
ae_model.put_on_device(device)
pred.put_on_device(device)


task_type = TBRR_SQP_DomainProvider.get_task_type()
solver_type = TBRR_SQP_DomainProvider.get_solver_type()
solver_config = TBRR_SQP_DomainProvider.get_solver_config()

task = task_type.sample(5)
assert pred.initial_solution is not None

results = []
solver = solver_type.init(solver_config)
for problem in task.export_problems():
    solver.setup(problem)
    result = solver.solve(pred.initial_solution)
    results.append(result)

desc_table = task.export_table()
lib = SolutionLibrary(  # FIXME: temp implemented this after deletion of model.infer
    task_type, solver_type, solver_config, ae_model, [pred], [0.0], [None], 1.0, "dummy"
)
ts = time.time()
lib.infer(task)
print(time.time() - ts)
# time to infer
