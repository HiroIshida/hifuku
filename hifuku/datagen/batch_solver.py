import logging
import os
import pickle
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import Process, get_context
from pathlib import Path
from typing import Generic, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import threadpoolctl
import tqdm
from skmp.solver.interface import AbstractScratchSolver, ConfigT, ResultT
from skmp.trajectory import Trajectory

from hifuku.config import ServerSpec
from hifuku.datagen.http_datagen.client import ClientBase
from hifuku.datagen.http_datagen.request import (
    SolveProblemRequest,
    http_connection,
    send_request,
)
from hifuku.datagen.utils import split_indices
from hifuku.pool import ProblemT
from hifuku.types import RawData
from hifuku.utils import filter_warnings

logger = logging.getLogger(__name__)

TrajectoryMaybeList = Union[List[Trajectory], Trajectory]


def duplicate_init_solution_if_not_list(
    init_solution: Optional[TrajectoryMaybeList], n_inner_task: int
) -> Sequence[Optional[Trajectory]]:
    init_solutions: Sequence[Optional[Trajectory]]
    if isinstance(init_solution, List):
        init_solutions = init_solution
    else:
        init_solutions = [init_solution] * n_inner_task
    return init_solutions


@dataclass
class DumpDatasetWorker(Generic[ProblemT, ConfigT, ResultT]):
    problems: List[ProblemT]
    solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]]
    solver_config: ConfigT
    init_solutions: Sequence[Optional[Trajectory]]
    results_list: List[Tuple[ResultT, ...]]
    cache_path: Path
    show_progress_bar: bool

    def __len__(self) -> int:
        return len(self.problems)

    def run(self):
        if self.show_progress_bar:
            logger.info("dump dataset")
        disable_progress_bar = not self.show_progress_bar
        for i in tqdm.tqdm(range(len(self)), disable=disable_progress_bar):
            problem = self.problems[i]
            init_solution = self.init_solutions[i]
            results = self.results_list[i]
            raw_data = RawData(init_solution, problem.export_table(), results, self.solver_config)
            name = str(uuid.uuid4()) + ".pkl"
            path = self.cache_path / name
            raw_data.dump(path)


class BatchProblemSolver(Generic[ConfigT, ResultT], ABC):
    solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]]
    config: ConfigT
    n_limit_batch: Optional[int]

    def __init__(
        self,
        solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]],
        config: ConfigT,
        n_limit_batch: Optional[int] = None,
    ):
        self.solver_t = solver_t
        self.config = config
        self.n_limit_batch = n_limit_batch

    def solve_batch(
        self,
        problems: List[ProblemT],
        init_solutions: Sequence[Optional[TrajectoryMaybeList]],
        use_default_solver: bool = False,
    ) -> List[Tuple[ResultT, ...]]:
        # When pickling-and-depickling, the procedure takes up much more memory than
        # pickled object size. I'm not sure but according to https://stackoverflow.com/a/38971446/7624196
        # the RAM usage could be two times bigger than the serialized-object size.
        # Thus, in the following, we first measure the pickle size, and splits the problem set
        # and then send the multiple chunk of problems sequentially.
        if self.n_limit_batch is None:
            max_ram_usage = 16 * 10**9
            problem_for_measuring = problems[0]
            problem_for_measuring.gridsdf  # access gridsdf because it's created lazily
            serialize_ram_size_each = len(pickle.dumps(problem_for_measuring)) * 2
            max_size = int(max_ram_usage // serialize_ram_size_each)
        else:
            max_size = self.n_limit_batch

        indices = range(len(problems))
        indices_list = np.array_split(indices, np.ceil(len(problems) / max_size))

        resultss = []
        for indices_part in indices_list:
            problems_part = [problems[i] for i in indices_part]
            init_solutions_est_list_part = [init_solutions[i] for i in indices_part]
            results_part = self._solve_batch_impl(problems_part, init_solutions_est_list_part, use_default_solver=use_default_solver)  # type: ignore
            resultss.extend(results_part)
        return resultss

    @abstractmethod
    def _solve_batch_impl(
        self,
        problems: List[ProblemT],
        init_solutions: Sequence[Optional[TrajectoryMaybeList]],
        use_default_solver: bool = False,
    ) -> List[Tuple[ResultT, ...]]:
        ...

    def dump_compressed_dataset_to_cachedir(
        self,
        problems: List[ProblemT],
        init_solutions: Sequence[TrajectoryMaybeList],
        cache_dir_path: Path,
        n_process: Optional[int],
    ) -> None:

        logger.debug("run self.solve_batch")
        results_list = self.solve_batch(problems, init_solutions)

        if n_process is None:
            cpu_count = os.cpu_count()
            assert cpu_count is not None
            n_process = int(0.5 * cpu_count)

        n_problem = len(problems)
        indices = np.array(list(range(n_problem)))
        indices_list = np.array_split(indices, n_process)

        logger.debug("dump results")
        process_list = []
        for i, indices_part in enumerate(indices_list):

            if len(indices_part) == 0:
                continue

            show_progress_bar = i == 0

            problems_part = [problems[i] for i in indices_part]
            init_solutions_part = [init_solutions[i] for i in indices_part]
            results_list_part = [results_list[i] for i in indices_part]

            worker = DumpDatasetWorker[ProblemT, ConfigT, ResultT](
                problems_part,
                self.solver_t,
                self.config,
                init_solutions_part,
                results_list_part,
                cache_dir_path,
                show_progress_bar,
            )
            p = Process(target=worker.run, args=())
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()


class MultiProcessBatchProblemSolver(BatchProblemSolver[ConfigT, ResultT]):
    n_process: int

    def __init__(
        self,
        solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]],
        config: ConfigT,
        n_process: Optional[int] = None,
        n_limit_batch: Optional[int] = None,
    ):
        super().__init__(solver_t, config, n_limit_batch)
        if n_process is None:
            logger.info("n_process is not set. automatically determine")
            cpu_num = os.cpu_count()
            assert cpu_num is not None
            n_process = int(cpu_num * 0.5)
        logger.info("n_process is set to {}".format(n_process))
        self.n_process = n_process

    def _solve_batch_impl(
        self,
        tasks: List[ProblemT],
        init_solutions: Sequence[Optional[TrajectoryMaybeList]],
        use_default_solver: bool = False,
    ) -> List[Tuple[ResultT, ...]]:

        filter_warnings()

        assert len(tasks) == len(init_solutions)
        assert len(tasks) > 0

        n_process = min(self.n_process, len(tasks))
        logger.info("*n_process: {}".format(n_process))
        logger.info("use_default_solver: {}".format(use_default_solver))

        is_single_process = n_process == 1
        if is_single_process:

            if use_default_solver:
                # NOTE: sovle_default does not return ResultT ...
                # Maybe we should replace ResultT with ResultProtocol ??
                return [tuple(task.solve_default()) for task in tasks]  # type: ignore
            else:
                results_list: List[Tuple[ResultT, ...]] = []
                n_max_call = self.config.n_max_call
                logger.debug("*n_max_call: {}".format(n_max_call))

                solver = self.solver_t.init(self.config)
                for task, init_solution in zip(tasks, init_solutions):
                    init_solutions_per_inner = duplicate_init_solution_if_not_list(
                        init_solution, task.n_inner_task
                    )
                    problems = task.export_problems()
                    results: List[ResultT] = []
                    for problem, init_solution_per_inner in zip(problems, init_solutions_per_inner):
                        solver.setup(problem)
                        result = solver.solve(init_solution_per_inner)
                        results.append(result)
                    results_list.append(tuple(results))
                return results_list
        else:
            # python's known bug when forking process while using logging module
            # https://stackoverflow.com/questions/65080123/python-multiprocessing-pool-some-process-in-deadlock-when-forked-but-runs-when-s
            # https://stackoverflow.com/questions/24509650/deadlock-with-logging-multiprocess-multithread-python-script
            # https://bugs.python.org/issue6721
            for hn in logger.handlers:
                assert hn.lock is not None
                assert not hn.lock.locked()

            args = []
            for idx in range(len(tasks)):
                args.append((idx, tasks[idx], init_solutions[idx]))

            with ProcessPoolExecutor(
                n_process,
                initializer=self._pool_setup,
                initargs=(self.solver_t, self.config, use_default_solver),
                mp_context=get_context("fork"),
            ) as executor:
                idx_results_pairs = list(
                    tqdm.tqdm(executor.map(self._pool_solve_single, args), total=len(args))
                )
            idx_results_pairs_sorted = sorted(idx_results_pairs, key=lambda x: x[0])  # type: ignore
            _, resultss = zip(*idx_results_pairs_sorted)
            return list(resultss)

    @staticmethod
    def _pool_setup(
        solver_t: Type[AbstractScratchSolver], config: ConfigT, use_default_solver: bool
    ):
        # NOTE: this function is used only in process pool
        # NOTE: a lot of type: ignore due to global variables
        global solver
        solver = solver_t.init(config)  # type: ignore
        global _use_default_solver
        _use_default_solver = use_default_solver  # type: ignore

    @staticmethod
    def _pool_solve_single(args):
        task_idx, task, init_solutions = args
        # NOTE: this function is used only in process pool
        # NOTE: a lot of type: ignore due to global variables
        has_gridsdf = task._gridsdf is not None

        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            global _use_default_solver
            if _use_default_solver:  # type: ignore
                results = task.solve_default()
            else:
                init_solutions = duplicate_init_solution_if_not_list(
                    init_solutions, task.n_inner_task
                )
                results = []
                for problem, init_solution in zip(task.export_problems(), init_solutions):
                    solver.setup(problem)  # type: ignore
                    result = solver.solve(init_solution)  # type: ignore
                    results.append(result)

            if has_gridsdf:
                task.invalidate_gridsdf()
        return task_idx, tuple(results)


HostPortPair = Tuple[str, int]


class DistributedBatchProblemSolver(
    ClientBase[SolveProblemRequest], BatchProblemSolver[ConfigT, ResultT]
):
    def __init__(
        self,
        solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]],
        config: ConfigT,
        server_specs: Optional[Tuple[ServerSpec, ...]] = None,
        use_available_host: bool = False,
        force_continue: bool = False,
        n_measure_sample: int = 40,
        n_limit_batch: Optional[int] = None,
    ):
        BatchProblemSolver.__init__(self, solver_t, config, n_limit_batch)
        ClientBase.__init__(
            self, server_specs, use_available_host, force_continue, n_measure_sample
        )

    @staticmethod  # called only in generate
    def send_and_recive_and_write(
        hostport: HostPortPair, request: SolveProblemRequest, indices: np.ndarray, tmp_path: Path
    ) -> None:

        logger.debug("send_and_recive_and_write called on pid: {}".format(os.getpid()))
        with http_connection(*hostport) as conn:
            response = send_request(conn, request)
        file_path = tmp_path / str(uuid.uuid4())
        with file_path.open(mode="wb") as f:
            pickle.dump((indices, response.results_list), f)
        logger.debug("saved to {}".format(file_path))
        logger.debug("send_and_recive_and_write finished on pid: {}".format(os.getpid()))

    def _solve_batch_impl(
        self,
        problems: List[ProblemT],
        init_solutions: Sequence[Optional[TrajectoryMaybeList]],
        use_default_solver: bool = False,
    ) -> List[Tuple[ResultT, ...]]:
        logger.debug("use_default_solver: {}".format(use_default_solver))

        hostport_pairs = list(self.hostport_cpuinfo_map.keys())
        problems_measure = problems[: self.n_measure_sample]
        init_solutions_measure = init_solutions[: self.n_measure_sample]
        request_for_measure = SolveProblemRequest(
            problems_measure,
            self.solver_t,
            self.config,
            init_solutions_measure,
            -1,
            use_default_solver,
        )
        n_problem = len(problems)
        n_problem_table = self.create_gen_number_table(request_for_measure, n_problem)
        indices_list = split_indices(n_problem, list(n_problem_table.values()))

        # send request
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            process_list = []
            for hostport, indices in zip(hostport_pairs, indices_list):
                n_process = self.hostport_cpuinfo_map[hostport].n_cpu
                problems_part = [problems[i] for i in indices]
                init_solutions_part = [init_solutions[i] for i in indices]
                req = SolveProblemRequest(
                    problems_part,
                    self.solver_t,
                    self.config,
                    init_solutions_part,
                    n_process,
                    use_default_solver,
                )
                if len(problems_part) > 0:
                    p = Process(
                        target=self.send_and_recive_and_write,
                        args=(hostport, req, indices, td_path),
                    )
                    p.start()
                    process_list.append(p)

            for p in process_list:
                p.join()

            results_list_all: List[List[ResultT]] = []
            indices_all: List[int] = []
            for file_path in td_path.iterdir():
                with file_path.open(mode="rb") as f:
                    try:
                        indices_part, results_list_part = pickle.load(f)
                    except Exception as e:
                        temp_file_path = "/tmp/malignant_pickle.pkl"
                        shutil.move(str(file_path), temp_file_path)
                        logger.error(
                            "loading pickle file failed. move the file at issue to {} for the future analysis."
                        )
                        raise e

                    results_list_all.extend(results_list_part)
                    indices_all.extend(indices_part)

            idx_result_pairs = list(zip(indices_all, results_list_all))
            idx_result_pairs_sorted = sorted(idx_result_pairs, key=lambda x: x[0])  # type: ignore
            _, results = zip(*idx_result_pairs_sorted)
            ret = list(results)
            assert len(ret) == len(problems)
        return ret  # type: ignore
