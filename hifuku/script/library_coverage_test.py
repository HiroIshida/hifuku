import numpy as np
import torch
from mohou.file import get_project_path

from hifuku.domain import TBRR_SQP_DomainProvider
from hifuku.library import SolutionLibrary
from hifuku.rpbench_wrap import TabletopBoxRightArmReachingTask

np.random.seed(0)

task_type = TBRR_SQP_DomainProvider.get_task_type()
solver_type = TBRR_SQP_DomainProvider.get_solver_type()
mp_batch_solver = TBRR_SQP_DomainProvider.get_multiprocess_batch_solver()
domain_name = TBRR_SQP_DomainProvider.get_domain_name()

pp = get_project_path("tabletop_solution_library-{}".format(domain_name))
validation_pool = [TabletopBoxRightArmReachingTask.sample(1) for _ in range(100)]
lib = SolutionLibrary.load(pp, task_type, solver_type, device=torch.device("cpu"))[0]
coverage = lib.measure_full_coverage(validation_pool, mp_batch_solver)
print(coverage)
