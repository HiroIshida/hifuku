import logging
import multiprocessing
import os
import pickle
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Generic, List, Optional, Tuple, Type

import numpy as np
import threadpoolctl
import tqdm
from skmp.solver.interface import AbstractScratchSolver, ConfigT, ResultT
from skmp.trajectory import Trajectory

from hifuku.config import ServerSpec
from hifuku.datagen.http_datagen.client import ClientBase
from hifuku.datagen.http_datagen.request import (
    SampleProblemRequest,
    SolveProblemRequest,
    http_connection,
    send_request,
)
from hifuku.pool import PredicatedProblemPool, ProblemT
from hifuku.types import RawData
from hifuku.utils import filter_warnings, num_torch_thread

logger = logging.getLogger(__name__)


def split_number(num, div):
    return [num // div + (1 if x < num % div else 0) for x in range(div)]


def split_indices(n_problem_total: int, n_problem_list: List[int]) -> List[List[int]]:
    indices = np.array(list(range(n_problem_total)))
    indices_list = []
    head = 0
    for n_problem in n_problem_list:
        tail = head + n_problem
        indices_list.append(indices[head:tail].tolist())
        head = tail
    assert sum([len(ii) for ii in indices_list]) == n_problem_total
    return indices_list


@dataclass
class BatchProblemSolverArg(Generic[ProblemT, ConfigT, ResultT]):
    indices: np.ndarray
    problems: List[ProblemT]
    solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]]
    solver_config: ConfigT
    init_solutions: List[Trajectory]
    show_process_bar: bool

    def __len__(self) -> int:
        return len(self.problems)

    def __post_init__(self) -> None:
        assert len(self.problems) == len(self.init_solutions)


class BatchProblemSolverWorker(Process, Generic[ProblemT, ConfigT, ResultT]):
    arg: BatchProblemSolverArg[ProblemT, ConfigT, ResultT]
    queue: Queue

    def __init__(self, arg: BatchProblemSolverArg, queue: Queue):
        self.arg = arg
        self.queue = queue
        super().__init__()

    def run(self) -> None:
        logger.debug("batch solver worker run with pid {}".format(os.getpid()))

        unique_id = (uuid.getnode() + os.getpid()) % (2**32 - 1)
        logger.debug("random seed set to {}".format(unique_id))

        np.random.seed(unique_id)
        disable_tqdm = not self.arg.show_process_bar

        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            # NOTE: because NLP solver and collision detection algrithm may use multithreading
            with tqdm.tqdm(total=len(self.arg), disable=disable_tqdm) as pbar:
                for idx, task, init_solution in zip(
                    self.arg.indices, self.arg.problems, self.arg.init_solutions
                ):
                    solver = self.arg.solver_t.init(self.arg.solver_config)

                    results = []
                    problems = task.export_problems()
                    for problem in tqdm.tqdm(problems, disable=disable_tqdm, leave=False):
                        solver.setup(problem)
                        result = solver.solve(init_solution)
                        results.append(result)
                    tupled_results = tuple(results)

                    logger.debug("generated single data")
                    logger.debug("success: {}".format([r.traj is not None for r in results]))
                    logger.debug("iteration: {}".format([r.n_call for r in results]))
                    self.queue.put((idx, tupled_results))
                    pbar.update(1)


@dataclass
class DumpDatasetWorker(Generic[ProblemT, ConfigT, ResultT]):
    problems: List[ProblemT]
    solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]]
    solver_config: ConfigT
    init_solutions: List[Trajectory]
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

    def __init__(self, solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]], config: ConfigT):
        self.solver_t = solver_t
        self.config = config

    @abstractmethod
    def solve_batch(
        self,
        problems: List[ProblemT],
        init_solutions: List[Trajectory],
    ) -> List[Tuple[ResultT, ...]]:
        ...

    def create_dataset(
        self,
        problems: List[ProblemT],
        init_solutions: List[Trajectory],
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
    ):
        super().__init__(solver_t, config)
        if n_process is None:
            logger.info("n_process is not set. automatically determine")
            cpu_num = os.cpu_count()
            assert cpu_num is not None
            n_process = int(cpu_num * 0.5)
        logger.info("n_process is set to {}".format(n_process))
        self.n_process = n_process

    def solve_batch(
        self,
        tasks: List[ProblemT],
        init_solutions: List[Trajectory],
    ) -> List[Tuple[ResultT, ...]]:

        filter_warnings()

        assert len(tasks) == len(init_solutions)
        assert len(tasks) > 0

        n_process = min(self.n_process, len(tasks))
        logger.debug("*n_process: {}".format(n_process))

        is_single_process = n_process == 1
        if is_single_process:
            results_list: List[Tuple[ResultT, ...]] = []
            n_max_call = self.config.n_max_call
            logger.debug("*n_max_call: {}".format(n_max_call))

            solver = self.solver_t.init(self.config)
            for task, init_solution in zip(tasks, init_solutions):
                results: List[ResultT] = []
                for problem in task.export_problems():
                    solver.setup(problem)
                    result = solver.solve(init_solution)
                    results.append(result)
                results_list.append(tuple(results))
            return results_list
        else:
            indices = np.array(list(range(len(tasks))))
            indices_list_per_worker = np.array_split(indices, n_process)

            q = multiprocessing.Queue()  # type: ignore
            indices_list_per_worker = np.array_split(indices, n_process)

            process_list = []
            for i, indices_part in enumerate(indices_list_per_worker):
                enable_tqdm = i == 0
                problems_part = [tasks[idx] for idx in indices_part]
                init_solutions_part = [init_solutions[idx] for idx in indices_part]
                arg = BatchProblemSolverArg(
                    indices_part,
                    problems_part,
                    self.solver_t,
                    self.config,
                    init_solutions_part,
                    enable_tqdm,
                )
                worker = BatchProblemSolverWorker(arg, q)  # type: ignore
                process_list.append(worker)
                worker.start()

            idx_result_pairs = [q.get() for _ in range(len(tasks))]

            for worker in process_list:
                worker.join()

            idx_result_pairs_sorted = sorted(idx_result_pairs, key=lambda x: x[0])  # type: ignore
            _, results = zip(*idx_result_pairs_sorted)
            return list(results)  # type: ignore


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
    ):
        BatchProblemSolver.__init__(self, solver_t, config)
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

    def solve_batch(
        self,
        problems: List[ProblemT],
        init_solutions: List[Trajectory],
    ) -> List[Tuple[ResultT, ...]]:

        hostport_pairs = list(self.hostport_cpuinfo_map.keys())
        problems_measure = problems[: self.n_measure_sample]
        init_solutions_measure = init_solutions[: self.n_measure_sample]
        request_for_measure = SolveProblemRequest(
            problems_measure, self.solver_t, self.config, init_solutions_measure, -1
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
                    problems_part, self.solver_t, self.config, init_solutions_part, n_process
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
                    indices_part, results_list_part = pickle.load(f)
                    results_list_all.extend(results_list_part)
                    indices_all.extend(indices_part)

            idx_result_pairs = list(zip(indices_all, results_list_all))
            idx_result_pairs_sorted = sorted(idx_result_pairs, key=lambda x: x[0])  # type: ignore
            _, results = zip(*idx_result_pairs_sorted)
        return list(results)  # type: ignore


class BatchProblemSampler(Generic[ProblemT], ABC):
    @abstractmethod
    def sample_batch(
        self,
        n_sample: int,
        pool: PredicatedProblemPool[ProblemT],
    ) -> List[ProblemT]:
        ...


class MultiProcessBatchProblemSampler(BatchProblemSampler[ProblemT]):
    n_process: int
    n_thread: int

    def __init__(self, n_process: Optional[int] = None):
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        n_physical_cpu = int(0.5 * cpu_count)

        if n_process is None:
            n_process = n_physical_cpu

        # if n_process is larger than physical core num
        # performance gain by parallelization is limited. so...
        n_process = min(n_process, n_physical_cpu)

        n_thread = n_physical_cpu // n_process

        logger.info("n_process is set to {}".format(n_process))
        logger.info("n_thread is set to {}".format(n_thread))
        assert (
            n_process * n_thread == n_physical_cpu
        ), "n_process: {}, n_thread: {}, n_physical_cpu {}".format(
            n_process, n_thread, n_physical_cpu
        )
        self.n_process = n_process
        self.n_thread = n_thread

    @staticmethod
    def work(
        n_sample: int,
        pool: PredicatedProblemPool[ProblemT],
        show_progress_bar: bool,
        n_thread: int,
        cache_path: Path,
    ) -> None:

        # set random seed
        unique_id = (uuid.getnode() + os.getpid()) % (2**32 - 1)
        np.random.seed(unique_id)
        logger.debug("random seed set to {}".format(unique_id))

        logger.debug("start sampling using clf")
        problems: List[ProblemT] = []

        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            # NOTE: numpy internal thread parallelization greatly slow down
            # the processing time when multuprocessing case, though the speed gain
            # by the thread parallelization is actually poor
            with num_torch_thread(n_thread):
                disable = not show_progress_bar
                with tqdm.tqdm(total=n_sample, smoothing=0.0, disable=disable) as pbar:
                    while len(problems) < n_sample:
                        problem = next(pool)
                        if problem is not None:
                            problems.append(problem)
                            pbar.update(1)
        ts = time.time()
        file_path = cache_path / str(uuid.uuid4())
        with file_path.open(mode="wb") as f:
            pickle.dump(problems, f)
        logger.debug("time to dump {}".format(time.time() - ts))

    def sample_batch(
        self,
        n_sample: int,
        pool: PredicatedProblemPool[ProblemT],
    ) -> List[ProblemT]:
        assert pool.parallelizable()
        assert n_sample > 0
        n_process = min(self.n_process, n_sample)

        with tempfile.TemporaryDirectory() as td:
            # spawn is safe but really slow.
            ctx = multiprocessing.get_context(method="fork")
            n_sample_list = split_number(n_sample, n_process)
            process_list = []

            td_path = Path(td)
            for idx_process, n_sample_part in enumerate(n_sample_list):
                show_progress = idx_process == 0
                args = (n_sample_part, pool, show_progress, self.n_thread, td_path)
                p = ctx.Process(target=self.work, args=args)  # type: ignore
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()
            logger.debug("finish all subprocess")

            ts = time.time()
            problems_sampled = []
            for file_path in td_path.iterdir():
                with file_path.open(mode="rb") as f:
                    problems_sampled.extend(pickle.load(f))
            logger.debug("time to load {}".format(time.time() - ts))
        return problems_sampled


class DistributeBatchProblemSampler(
    ClientBase[SampleProblemRequest], BatchProblemSampler[ProblemT]
):
    @staticmethod  # called only in generate
    def send_and_recive_and_write(
        hostport: HostPortPair, request: SampleProblemRequest, tmp_path: Path
    ) -> None:
        logger.debug("send_and_recive_and_write called on pid: {}".format(os.getpid()))
        with http_connection(*hostport) as conn:
            response = send_request(conn, request)
        file_path = tmp_path / str(uuid.uuid4())
        assert len(response.problems) > 0
        with file_path.open(mode="wb") as f:
            pickle.dump((response.problems), f)
        logger.debug("saved to {}".format(file_path))
        logger.debug("send_and_recive_and_write finished on pid: {}".format(os.getpid()))

    def sample_batch(
        self,
        n_sample: int,
        pool: PredicatedProblemPool[ProblemT],
    ) -> List[ProblemT]:
        assert n_sample > 0
        assert pool.parallelizable()

        hostport_pairs = list(self.hostport_cpuinfo_map.keys())
        request_for_measure = SampleProblemRequest(self.n_measure_sample, pool, -1)
        n_sample_table = self.create_gen_number_table(request_for_measure, n_sample)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            process_list = []
            for hostport in hostport_pairs:
                n_sample_part = n_sample_table[hostport]
                if n_sample_part > 0:
                    n_process = self.hostport_cpuinfo_map[hostport].n_cpu
                    req = SampleProblemRequest(n_sample_part, pool, n_process)

                    p = Process(
                        target=self.send_and_recive_and_write, args=(hostport, req, td_path)
                    )
                    p.start()
                    process_list.append(p)

            for p in process_list:
                p.join()

            problems = []
            for file_path in td_path.iterdir():
                with file_path.open(mode="rb") as f:
                    problems_part = pickle.load(f)
                    problems.extend(problems_part)
        return problems
