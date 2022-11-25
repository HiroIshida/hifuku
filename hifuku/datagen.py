import logging
import math
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
from typing import ClassVar, Dict, Generic, List, Optional, Tuple

import dill
import numpy as np
import tqdm

from hifuku.http_datagen.client import ClientBase
from hifuku.http_datagen.request import (
    GetCPUInfoResponse,
    SolveProblemRequest,
    http_connection,
    send_request,
)
from hifuku.pool import PredicatedIteratorProblemPool
from hifuku.types import ProblemT, RawData, ResultProtocol
from hifuku.utils import num_torch_thread

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
class BatchProblemSolverArg(Generic[ProblemT]):
    indices: np.ndarray
    problems: List[ProblemT]
    init_solutions: List[np.ndarray]
    show_process_bar: bool

    def __len__(self) -> int:
        return len(self.problems)

    def __post_init__(self) -> None:
        assert len(self.problems) == len(self.init_solutions)


class BatchProblemSolverTask(Process, Generic[ProblemT]):
    arg: BatchProblemSolverArg[ProblemT]
    queue: Queue

    def __init__(self, arg: BatchProblemSolverArg, queue: Queue):
        self.arg = arg
        self.queue = queue
        super().__init__()

    def run(self) -> None:
        logger.debug("DataGenerationTask.run with pid {}".format(os.getpid()))

        unique_id = (uuid.getnode() + os.getpid()) % (2**32 - 1)
        logger.debug("random seed set to {}".format(unique_id))

        np.random.seed(unique_id)
        disable_tqdm = not self.arg.show_process_bar

        with tqdm.tqdm(total=len(self.arg), disable=disable_tqdm) as pbar:
            for idx, prob, init_solution in zip(
                self.arg.indices, self.arg.problems, self.arg.init_solutions
            ):
                results = prob.solve(init_solution)
                logger.debug("generated single data")
                logger.debug("success: {}".format([r.success for r in results]))
                logger.debug("iteration: {}".format([r.nit for r in results]))
                self.queue.put((idx, results))
                pbar.update(1)


@dataclass
class DumpResultTask(Generic[ProblemT]):
    problems: List[ProblemT]
    init_solutions: List[np.ndarray]
    results_list: List[Tuple[ResultProtocol, ...]]
    cache_path: Path

    def __len__(self) -> int:
        return len(self.problems)

    def run(self):
        for i in range(len(self)):
            problem = self.problems[i]
            init_solution = self.init_solutions[i]
            results = self.results_list[i]
            raw_data = RawData.create(problem, results, init_solution)
            name = str(uuid.uuid4()) + ".npz"
            path = self.cache_path / name
            raw_data.dump(path)


class BatchProblemSolver(Generic[ProblemT], ABC):
    @abstractmethod
    def solve_batch(
        self,
        problems: List[ProblemT],
        init_solutions: List[np.ndarray],
    ) -> List[Tuple[ResultProtocol, ...]]:
        ...

    def create_dataset(
        self,
        problems: List[ProblemT],
        init_solutions: List[np.ndarray],
        cache_dir_path: Path,
        n_process: Optional[int],
    ) -> None:

        results_list = self.solve_batch(problems, init_solutions)

        if n_process is None:
            cpu_count = os.cpu_count()
            assert cpu_count is not None
            n_process = int(0.5 * cpu_count)

        n_problem = len(problems)
        indices = np.array(list(range(n_problem)))
        indices_list = np.array_split(indices, n_process)

        process_list = []
        for indices_part in indices_list:
            problems_part = [problems[i] for i in indices_part]
            init_solutions_part = [init_solutions[i] for i in indices_part]
            results_list_part = [results_list[i] for i in indices_part]

            task = DumpResultTask(
                problems_part, init_solutions_part, results_list_part, cache_dir_path
            )
            p = Process(target=task.run, args=())
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()


class MultiProcessBatchProblemSolver(BatchProblemSolver[ProblemT]):
    n_process: int

    def __init__(self, n_process: Optional[int] = None):
        if n_process is None:
            logger.info("n_process is not set. automatically determine")
            cpu_num = os.cpu_count()
            assert cpu_num is not None
            n_process = int(cpu_num * 0.5)
        logger.info("n_process is set to {}".format(n_process))
        self.n_process = n_process

    def solve_batch(
        self,
        problems: List[ProblemT],
        init_solutions: List[np.ndarray],
    ) -> List[Tuple[ResultProtocol, ...]]:

        assert len(problems) == len(init_solutions)
        assert len(problems) > 0

        n_process = min(self.n_process, len(problems))
        logger.debug("*n_process: {}".format(n_process))

        is_single_process = n_process == 1
        if is_single_process:
            results_list: List[Tuple[ResultProtocol, ...]] = []
            maxiter = problems[0].get_solver_config().maxiter
            logger.debug("*maxiter: {}".format(maxiter))
            for problem, init_solution in zip(problems, init_solutions):
                results = problem.solve(init_solution)
                results_list.append(results)
            return results_list
        else:
            indices = np.array(list(range(len(problems))))
            indices_list_per_worker = np.array_split(indices, n_process)

            q = multiprocessing.Queue()  # type: ignore
            indices_list_per_worker = np.array_split(indices, n_process)

            process_list = []
            for i, indices_part in enumerate(indices_list_per_worker):
                enable_tqdm = i == 0
                problems_part = [problems[idx] for idx in indices_part]
                init_solutions_part = [init_solutions[idx] for idx in indices_part]
                arg = BatchProblemSolverArg(
                    indices_part, problems_part, init_solutions_part, enable_tqdm
                )
                task = BatchProblemSolverTask(arg, q)  # type: ignore
                process_list.append(task)
                task.start()

            idx_result_pairs = [q.get() for _ in range(len(problems))]

            for task in process_list:
                task.join()

            idx_result_pairs_sorted = sorted(idx_result_pairs, key=lambda x: x[0])  # type: ignore
            _, results = zip(*idx_result_pairs_sorted)
            return list(results)  # type: ignore


HostPortPair = Tuple[str, int]


class DistributedBatchProblemSolver(ClientBase[SolveProblemRequest], BatchProblemSolver[ProblemT]):
    hostport_cpuinfo_map: Dict[HostPortPair, GetCPUInfoResponse]
    check_module_names: ClassVar[Tuple[str, ...]] = ("skplan", "voxbloxpy")

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
        init_solutions: List[np.ndarray],
    ) -> List[Tuple[ResultProtocol, ...]]:

        hostport_pairs = list(self.hostport_cpuinfo_map.keys())
        problems_measure = problems[: self.n_measure_sample]
        init_solutions_measure = init_solutions[: self.n_measure_sample]
        performance_table = self._measure_performance_of_each_server(
            problems_measure, init_solutions_measure
        )
        logger.info("performance table: {}".format(performance_table))

        n_problem_table: Dict[HostPortPair, int] = {}
        n_problem = len(problems)
        for hostport in hostport_pairs:
            n_problem_host = math.floor(n_problem * performance_table[hostport])
            n_problem_table[hostport] = n_problem_host

        # allocate remainders
        remainder_sum = n_problem - sum(n_problem_table.values())
        alloc_splitted = split_number(remainder_sum, len(hostport_pairs))
        for hostport, alloc in zip(hostport_pairs, alloc_splitted):
            n_problem_table[hostport] += alloc

        assert sum(n_problem_table.values()) == n_problem
        logger.info("n_problem_table: {}".format(n_problem_table))

        indices_list = split_indices(n_problem, list(n_problem_table.values()))

        # send request
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            process_list = []
            for hostport, indices in zip(hostport_pairs, indices_list):
                n_process = self.hostport_cpuinfo_map[hostport].n_cpu
                problems_part = [problems[i] for i in indices]
                init_solutions_part = [init_solutions[i] for i in indices]
                req = SolveProblemRequest(problems_part, init_solutions_part, n_process)
                if len(problems_part) > 0:
                    p = Process(
                        target=self.send_and_recive_and_write,
                        args=(hostport, req, indices, td_path),
                    )
                    p.start()
                    process_list.append(p)

            for p in process_list:
                p.join()

            results_list_all: List[List[ResultProtocol]] = []
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

    @staticmethod  # called only in _measure_performance_of_each_server
    def _send_and_recive_and_get_elapsed_time(
        hostport: HostPortPair, request: SolveProblemRequest, queue: Queue
    ) -> None:
        logger.debug("send_and_recive_and_get_elapsed_time called on pid: {}".format(os.getpid()))
        with http_connection(*hostport) as conn:
            response = send_request(conn, request)
        logger.debug("send_and_recive_and_get_elapsed_time finished on pid: {}".format(os.getpid()))
        queue.put((hostport, response.elapsed_time))

    def _measure_performance_of_each_server(
        self,
        problems: List[ProblemT],
        init_solutions: List[np.ndarray],
    ) -> Dict[HostPortPair, float]:

        logger.info("measure performance of each server by letting them make a dummy dataset")
        score_map: Dict[HostPortPair, float] = {}
        with tempfile.TemporaryDirectory() as td:
            Path(td)
            queue = Queue()  # type: ignore
            process_list = []
            for hostport in self.hostport_cpuinfo_map.keys():
                cpu_info = self.hostport_cpuinfo_map[hostport]
                req = SolveProblemRequest(problems, init_solutions, cpu_info.n_cpu)
                p = Process(
                    target=self._send_and_recive_and_get_elapsed_time, args=(hostport, req, queue)
                )
                process_list.append(p)
                p.start()

            hostport_elapsed_pairs = [queue.get() for _ in range(len(self.hostport_cpuinfo_map))]
            for p in process_list:
                p.join()

        for hostport, elapsed in hostport_elapsed_pairs:
            score_map[hostport] = 1.0 / elapsed

        # normalize
        score_sum = sum(score_map.values())
        for key in score_map:
            score_map[key] /= score_sum
        return score_map


class BatchProblemSampler(Generic[ProblemT], ABC):
    @abstractmethod
    def sample_batch(
        self,
        n_sample: int,
        pool: PredicatedIteratorProblemPool[ProblemT],
    ) -> List[ProblemT]:
        ...


class MultiProcessBatchProblemSampler(BatchProblemSampler[ProblemT]):
    n_process: int
    n_thread: int

    def __init__(self, n_process: Optional[int] = None, n_thread: int = 1):
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        n_physical_cpu = int(0.5 * cpu_count)

        if n_process is None:
            good_thread_num = 2  # from my experience
            n_process = n_physical_cpu // good_thread_num
        logger.info("n_process is set to {}".format(n_process))
        logger.info("n_thread is set to {}".format(n_thread))
        assert n_process * n_thread == n_physical_cpu  # hmm, too strict
        self.n_process = n_process
        self.n_thread = n_thread

    @staticmethod
    def task(
        n_sample: int,
        pool: PredicatedIteratorProblemPool[ProblemT],
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
            dill.dump(problems, f)
        logger.debug("time to dump {}".format(time.time() - ts))

    def sample(
        self,
        n_sample: int,
        pool: PredicatedIteratorProblemPool[ProblemT],
    ) -> List[ProblemT]:

        assert n_sample > self.n_process * 5  # this is random. i don't have time

        with tempfile.TemporaryDirectory() as td:
            # https://github.com/pytorch/pytorch/issues/89693
            ctx = multiprocessing.get_context(method="spawn")
            n_sample_list = split_number(n_sample, self.n_process)
            process_list = []

            td_path = Path(td)
            for idx_process, n_sample_part in enumerate(n_sample_list):
                show_progress = idx_process == 0
                args = (n_sample_part, pool, show_progress, self.n_thread, td_path)
                p = ctx.Process(target=self.task, args=args)
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()

            ts = time.time()
            problems_sampled = []
            for file_path in td_path.iterdir():
                with file_path.open(mode="rb") as f:
                    problems_sampled.extend(dill.load(f))
            logger.debug("time to load {}".format(time.time() - ts))
        return problems_sampled
