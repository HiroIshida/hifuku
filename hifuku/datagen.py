import logging
import math
import multiprocessing
import os
import pickle
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import ClassVar, Dict, Generic, List, Optional, Tuple, Type

import numpy as np
import tqdm

from hifuku.http_datagen.request import (
    GetCPUInfoRequest,
    GetCPUInfoResponse,
    GetModuleHashValueRequest,
    SolveProblemRequest,
    http_connection,
    send_request,
)
from hifuku.types import ProblemT, ResultProtocol
from hifuku.utils import get_module_source_hash

logger = logging.getLogger(__name__)


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


class BatchProblemSolver(Generic[ProblemT], ABC):
    problem_type: Type[ProblemT]

    def __init__(self, problem_type: Type[ProblemT], cache_base_dir: Optional[Path] = None):
        self.problem_type = problem_type

    @abstractmethod
    def generate(
        self,
        problems: List[ProblemT],
        init_solutions: List[np.ndarray],
    ) -> List[Tuple[ResultProtocol, ...]]:
        ...

    @staticmethod
    def split_number(num, div):
        return [num // div + (1 if x < num % div else 0) for x in range(div)]

    @staticmethod
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


class MultiProcessBatchProblemSolver(BatchProblemSolver[ProblemT]):
    n_process: int

    def __init__(self, problem_type: Type[ProblemT], n_process: Optional[int] = None):
        super().__init__(problem_type)
        if n_process is None:
            logger.info("n_process is not set. automatically determine")
            cpu_num = os.cpu_count()
            assert cpu_num is not None
            n_process = int(cpu_num * 0.5)
        logger.info("n_process is set to {}".format(n_process))
        self.n_process = n_process

    def generate(
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


class DistributedBatchProblemSolver(BatchProblemSolver[ProblemT]):
    hostport_cpuinfo_map: Dict[HostPortPair, GetCPUInfoResponse]
    n_problem_measure: int
    check_module_names: ClassVar[Tuple[str, ...]] = ("skplan", "voxbloxpy")

    def __init__(
        self,
        problem_type: Type[ProblemT],
        host_port_pairs: List[HostPortPair],
        use_available_host: bool = False,
        force_continue: bool = False,
        n_problem_measure: int = 40,
    ):
        super().__init__(problem_type)
        self.hostport_cpuinfo_map = self._init_get_cpu_infos(host_port_pairs, use_available_host)
        list(self.hostport_cpuinfo_map.keys())
        # self._init_check_dependent_module_hash(available_hostport_pairs, force_continue)
        self.n_problem_measure = n_problem_measure

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

    def generate(
        self,
        problems: List[ProblemT],
        init_solutions: List[np.ndarray],
    ) -> List[Tuple[ResultProtocol, ...]]:

        hostport_pairs = list(self.hostport_cpuinfo_map.keys())
        problems_measure = problems[: self.n_problem_measure]
        init_solutions_measure = init_solutions[: self.n_problem_measure]
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
        alloc_splitted = self.split_number(remainder_sum, len(hostport_pairs))
        for hostport, alloc in zip(hostport_pairs, alloc_splitted):
            n_problem_table[hostport] += alloc

        assert sum(n_problem_table.values()) == n_problem
        logger.info("n_problem_table: {}".format(n_problem_table))

        indices_list = self.split_indices(n_problem, list(n_problem_table.values()))

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

    @classmethod  # called only one in __init__
    def _init_get_cpu_infos(
        cls, host_port_pairs: List[HostPortPair], use_available_host: bool
    ) -> Dict[HostPortPair, GetCPUInfoResponse]:
        # this method also work as connection checker
        logger.info("check connection by cpu request")
        hostport_cpuinfo_map: Dict[Tuple[str, int], GetCPUInfoResponse] = {}
        for host, port in host_port_pairs:
            try:
                with http_connection(host, port) as conn:
                    req_cpu = GetCPUInfoRequest()
                    resp_cpu = send_request(conn, req_cpu)
                logger.info("cpu info of ({}, {}) is {}".format(host, port, resp_cpu))
                hostport_cpuinfo_map[(host, port)] = resp_cpu
            except ConnectionRefusedError:
                logger.error("connection to ({}, {}) was refused ".format(host, port))

        if not use_available_host:
            if len(hostport_cpuinfo_map) != len(host_port_pairs):
                logger.error("connection to some of the specified hosts are failed")
                raise ConnectionRefusedError
        return hostport_cpuinfo_map

    @classmethod  # called only one in __init__
    def _init_check_dependent_module_hash(
        cls, hostport_pairs: List[HostPortPair], force_continue: bool
    ):
        logger.info("check dependent module hash matches to the client ones")
        invalid_pairs = []
        hash_list_client = [get_module_source_hash(name) for name in cls.check_module_names]
        logger.debug("hash value client: {}".format(hash_list_client))
        for host, port in hostport_pairs:
            with http_connection(host, port) as conn:
                req_hash = GetModuleHashValueRequest(list(cls.check_module_names))
                resp_hash = send_request(conn, req_hash)
                if resp_hash.hash_values != hash_list_client:
                    invalid_pairs.append((host, port))

        if len(invalid_pairs) > 0:
            message = "hosts {} module is incompatble with client ones".format(invalid_pairs)
            logger.error(message)
            if not force_continue:
                raise RuntimeError(message)

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
