import logging
import math
import os
import tempfile
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from pathlib import Path
from typing import ClassVar, Dict, Generic, List, Optional, Tuple, Type

import numpy as np

from hifuku.http_datagen.request import (
    CreateDatasetRequest,
    GetCPUInfoRequest,
    GetCPUInfoResponse,
    GetModuleHashValueRequest,
    http_connection,
    send_request,
)
from hifuku.llazy.generation import DataGenerationTask, DataGenerationTaskArg
from hifuku.types import ProblemInterface, ProblemT, RawData
from hifuku.utils import get_module_source_hash

logger = logging.getLogger(__name__)


class HifukuDataGenerationTask(DataGenerationTask[RawData]):
    problem_type: Type[ProblemInterface]
    n_problem_inner: int
    init_solution: np.ndarray

    def __init__(
        self,
        arg: DataGenerationTaskArg,
        problem_type: Type[ProblemInterface],
        n_problem_inner: int,
        init_solution: np.ndarray,
    ):
        super().__init__(arg)
        self.problem_type = problem_type
        self.n_problem_inner = n_problem_inner
        self.init_solution = init_solution

    def post_init_hook(self) -> None:
        pass

    def generate_single_data(self) -> RawData:
        problem = self.problem_type.sample(self.n_problem_inner)
        results = problem.solve(self.init_solution)
        logger.debug("generated single data")
        logger.debug("success: {}".format([r.success for r in results]))
        logger.debug("iteration: {}".format([r.nit for r in results]))
        data = RawData.create(problem, results, self.init_solution)
        return data


class DatasetGenerator(Generic[ProblemT], ABC):
    problem_type: Type[ProblemT]

    def __init__(self, problem_type: Type[ProblemT], cache_base_dir: Optional[Path] = None):
        self.problem_type = problem_type

    @abstractmethod
    def generate(
        self, init_solution: np.ndarray, n_problem: int, n_problem_inner, cache_dir_path: Path
    ) -> None:
        pass

    @staticmethod
    def split_number(num, div):
        return [num // div + (1 if x < num % div else 0) for x in range(div)]


class MultiProcessDatasetGenerator(DatasetGenerator[ProblemT]):
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
        self, init_solution: np.ndarray, n_problem: int, n_problem_inner, cache_dir_path: Path
    ) -> None:
        n_problem_per_process_list = self.split_number(n_problem, self.n_process)
        assert cache_dir_path.exists()

        if self.n_process > 1:
            process_list = []
            for idx_process, n_problem_per_process in enumerate(n_problem_per_process_list):
                show_process_bar = idx_process == 1
                arg = DataGenerationTaskArg(
                    n_problem_per_process, show_process_bar, cache_dir_path, extension=".npz"
                )
                p = HifukuDataGenerationTask(arg, self.problem_type, n_problem_inner, init_solution)
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()
        else:
            arg = DataGenerationTaskArg(n_problem, True, cache_dir_path, extension=".npz")
            task = HifukuDataGenerationTask(arg, self.problem_type, n_problem_inner, init_solution)
            task.run()


HostPortPair = Tuple[str, int]


class DistributedDatasetGenerator(DatasetGenerator[ProblemT]):
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
        hostport: HostPortPair, request: CreateDatasetRequest, cache_dir_path: Path
    ) -> None:
        logger.debug("send_and_recive_and_write called on pid: {}".format(os.getpid()))
        with http_connection(*hostport) as conn:
            response = send_request(conn, request)
        for data, file_name in zip(response.data_list, response.name_list):
            file_path = cache_dir_path / file_name
            with file_path.open(mode="wb") as f:
                f.write(data)
        logger.debug("send_and_recive_and_write finished on pid: {}".format(os.getpid()))

    def generate(
        self, init_solution: np.ndarray, n_problem: int, n_problem_inner, cache_dir_path: Path
    ) -> None:
        hostport_pairs = list(self.hostport_cpuinfo_map.keys())
        performance_table = self._measure_performance_of_each_server(
            self.n_problem_measure, n_problem_inner
        )
        logger.info("performance table: {}".format(performance_table))

        n_problem_table: Dict[HostPortPair, int] = {}
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

        # send request
        process_list = []
        for hostport in hostport_pairs:
            n_problem_host = n_problem_table[hostport]
            n_process = self.hostport_cpuinfo_map[hostport].n_cpu
            req = CreateDatasetRequest(
                self.problem_type, init_solution, n_problem_host, n_problem_inner, n_process
            )
            p = Process(target=self.send_and_recive_and_write, args=(hostport, req, cache_dir_path))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()

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
        hostport: HostPortPair, request: CreateDatasetRequest, queue: Queue
    ) -> None:
        logger.debug("send_and_recive_and_get_elapsed_time called on pid: {}".format(os.getpid()))
        with http_connection(*hostport) as conn:
            response = send_request(conn, request)
        logger.debug("send_and_recive_and_get_elapsed_time finished on pid: {}".format(os.getpid()))
        queue.put((hostport, response.elapsed_time))

    def _measure_performance_of_each_server(
        self,
        n_problem: int,
        n_problem_inner: int,
    ) -> Dict[HostPortPair, float]:
        n_max_trial = 10
        count = 0
        init_solution: Optional[np.ndarray] = None
        problem_standard = self.problem_type.create_standard()
        while True:
            try:
                logger.debug("try solving standard problem...")
                result = problem_standard.solve()[0]
                if result.success:
                    logger.debug("solved!")
                    init_solution = result.x
                    break
            except self.problem_type.SamplingBasedInitialguessFail:
                pass
            count += 1
            if count > n_max_trial:
                raise RuntimeError("somehow standard problem cannot be solved")
        assert init_solution is not None

        logger.info("measure performance of each server by letting them make a dummy dataset")
        score_map: Dict[HostPortPair, float] = {}
        with tempfile.TemporaryDirectory() as td:
            Path(td)
            queue = Queue()  # type: ignore
            process_list = []
            for hostport in self.hostport_cpuinfo_map.keys():
                cpu_info = self.hostport_cpuinfo_map[hostport]
                req = CreateDatasetRequest(
                    self.problem_type,
                    init_solution,
                    n_problem=n_problem,
                    n_problem_inner=n_problem_inner,
                    n_process=cpu_info.n_cpu,
                )
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
