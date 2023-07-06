import warnings

import numpy as np
from mohou.file import get_project_path

from hifuku.domain import TORR_SQP_Domain

warnings.filterwarnings("ignore", message="Values in x were outside bounds during")
warnings.filterwarnings("ignore", message="texture specified in URDF is not supported")

np.random.seed(0)

if __name__ == "__main__":
    n_problem = 100
    pp = get_project_path("TORR_SQP")
    pp.mkdir(exist_ok=True)
    cache_base_path = pp / "cache"
    cache_base_path.mkdir(exist_ok=True)
    batch_solver = TORR_SQP_Domain.get_multiprocess_batch_solver()

    std_task = TORR_SQP_Domain.task_type.sample(1, standard=True)
    init_solution = std_task.solve_default()[0].traj
    assert init_solution is not None
    problems = [TORR_SQP_Domain.task_type.sample(50) for _ in range(n_problem)]
    init_solutions = [init_solution] * n_problem
    batch_solver.dump_compressed_dataset_to_cachedir(
        problems, init_solutions, cache_base_path, None
    )
