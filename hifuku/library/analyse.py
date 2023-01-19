import numpy as np
import tqdm
from skmp.solver.interface import ConfigT, ResultT

from hifuku.library.experimental import SolutionLibrary
from hifuku.pool import ProblemT


def compute_adj_matrix(
    lib: SolutionLibrary[ProblemT, ConfigT, ResultT], n_sample: int
) -> np.ndarray:
    lib.success_iter_threshold

    bools_list = []
    for _ in tqdm.tqdm(range(n_sample)):
        task = lib.task_type.sample(1)
        iters = lib._infer_iteration_num(task)
        bools = iters < lib.success_iter_threshold()
        bools_list.append(bools)
    bools_mat = np.array(bools_list)

    n_element = len(lib.predictors)
    adj_mat = np.zeros((n_element, n_element), dtype=bool)
    for i in range(n_element):
        for j in range(n_element):
            if i == j:
                continue
            is_connected = np.any(np.logical_and(bools_mat[i], bools_mat[j]))
            adj_mat[i, j] = is_connected
    return adj_mat
