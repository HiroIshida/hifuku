import time

import torch
from mohou.file import get_project_path
from mohou.trainer import TrainCache

from hifuku.domain import TBRR_SQP_Domain
from hifuku.library.core import SolutionLibrary
from hifuku.neuralnet import IterationPredictor
from hifuku.script_utils import load_compatible_autoencoder

ae_model = load_compatible_autoencoder(TBRR_SQP_Domain)

pp = get_project_path("TBRR_SQP")
pred: IterationPredictor = TrainCache.load_latest(pp, IterationPredictor).best_model

device = torch.device("cpu")
ae_model.put_on_device(device)
pred.put_on_device(device)


task_type = TBRR_SQP_Domain.task_type

task = task_type.sample(5)
assert pred.initial_solution is not None

results = []
solver = TBRR_SQP_Domain.create_solver()
for problem in task.export_problems():
    solver.setup(problem)
    result = solver.solve(pred.initial_solution)
    results.append(result)

desc_table = task.export_table()
lib = SolutionLibrary(  # FIXME: temp implemented this after deletion of model.infer
    task_type,
    TBRR_SQP_Domain.solver_type,
    TBRR_SQP_Domain.solver_config,
    ae_model,
    [pred],
    [0.0],
    [None],
    1.0,
    "dummy",
)
ts = time.time()
lib.infer(task)
print(time.time() - ts)
# time to infer
