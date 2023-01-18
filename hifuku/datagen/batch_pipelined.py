from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Generic, List

import numpy as np
import threadpoolctl
import tqdm
from skmp.solver.interface import ResultProtocol, ResultT

from hifuku.datagen.utils import split_number
from hifuku.pool import PredicatedProblemPool, ProblemT
from hifuku.utils import num_torch_thread


@dataclass
class WorkerOutput(Generic[ProblemT, ResultT]):
    task: ProblemT
    results: List[ResultT]


@dataclass
class WorkerArg(Generic[ProblemT]):
    n_sample: int
    pool: PredicatedProblemPool[ProblemT]
    show_progress_bar: bool
    queue: "Queue[WorkerOutput[ProblemT, ResultProtocol]]"


def work(arg: WorkerArg[ProblemT]) -> None:

    count = 0

    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        # NOTE: numpy internal thread parallelization greatly slow down
        # the processing time when multuprocessing case, though the speed gain
        # by the thread parallelization is actually poor
        with num_torch_thread(1):
            disable = not arg.show_progress_bar
            with tqdm.tqdm(total=arg.n_sample, smoothing=0.0, disable=disable) as pbar:

                while count < arg.n_sample:
                    task = next(arg.pool)
                    if task is not None:

                        results = []

                        results = task.solve_default()
                        is_valid = np.all([res.traj is not None for res in results])
                        if is_valid:
                            output = WorkerOutput(task, results)
                            arg.queue.put(output)
                            count += 1
                            pbar.update(1)


def sample_feasible_problem_with_solution(
    n_sample: int, pool: PredicatedProblemPool[ProblemT], n_process: int
) -> List[WorkerOutput]:

    n_sample_list = split_number(n_sample, n_process)
    queue: "Queue[WorkerOutput[ProblemT, ResultProtocol]]" = Queue()
    processes = []
    for i, n_sample_partial in enumerate(n_sample_list):
        show_progress_bar = i == 0
        arg = WorkerArg(n_sample_partial, pool, show_progress_bar, queue)
        p = Process(target=work, args=(arg,))
        p.start()
        processes.append(p)

    outputs = [queue.get() for _ in range(n_sample)]

    for p in processes:
        p.join()
    return outputs
