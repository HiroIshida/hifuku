import logging
import multiprocessing
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import tqdm

from hifuku.types import ProblemInterface

logger = logging.getLogger(__name__)


@dataclass
class ComputeRealItervalsArg:
    indices: np.ndarray
    problems: Sequence[ProblemInterface]
    init_solution: np.ndarray
    maxiter: int
    disable_tqdm: bool


def _compute_real_itervals(arg: ComputeRealItervalsArg, q: multiprocessing.Queue):
    with tqdm.tqdm(total=len(arg.problems), disable=arg.disable_tqdm) as pbar:
        for idx, problem in zip(arg.indices, arg.problems):
            assert problem.n_problem() == 1
            result = problem.solve(arg.init_solution)[0]
            iterval_real = result.nit if result.success else float(arg.maxiter)
            q.put((idx, iterval_real))
            pbar.update(1)


def compute_real_itervals(
    problems: Sequence[ProblemInterface],
    init_solution: np.ndarray,
    maxiter: int,
    n_process: Optional[int] = None,
) -> List[float]:
    if n_process is None:
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        n_process = int(0.5 * cpu_count)

    is_single_process = n_process == 1
    if is_single_process:
        itervals = []
        for problem in problems:
            result = problem.solve(init_solution)[0]
            iterval_real = result.nit if result.success else float(maxiter)
            itervals.append(iterval_real)
        return itervals
    else:
        indices = np.array(list(range(len(problems))))
        indices_list_per_worker = np.array_split(indices, n_process)

        q = multiprocessing.Queue()  # type: ignore
        indices_list_per_worker = np.array_split(indices, n_process)

        process_list = []
        for i, indices_part in enumerate(indices_list_per_worker):
            disable_tqdm = i > 0
            problems_part = [problems[idx] for idx in indices_part]
            arg = ComputeRealItervalsArg(
                indices_part, problems_part, init_solution, maxiter, disable_tqdm
            )
            p = multiprocessing.Process(target=_compute_real_itervals, args=(arg, q))
            p.start()
            process_list.append(p)

        idx_iterval_pairs = [q.get() for _ in range(len(problems))]
        idx_iterval_pairs_sorted = sorted(idx_iterval_pairs, key=lambda x: x[0])  # type: ignore
        _, itervals = zip(*idx_iterval_pairs_sorted)
        return list(itervals)
