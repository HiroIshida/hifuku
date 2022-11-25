import logging
import os
import tempfile
from multiprocessing import Process, Queue
from pathlib import Path
from typing import ClassVar, Dict, Generic, List, Tuple

from hifuku.http_datagen.request import (
    GetCPUInfoRequest,
    GetCPUInfoResponse,
    GetModuleHashValueRequest,
    MainRequestT,
    http_connection,
    send_request,
)
from hifuku.utils import get_module_source_hash

logger = logging.getLogger(__name__)


HostPortPair = Tuple[str, int]


class ClientBase(Generic[MainRequestT]):
    hostport_cpuinfo_map: Dict[HostPortPair, GetCPUInfoResponse]
    n_measure_sample: int
    check_module_names: ClassVar[Tuple[str, ...]] = ("skplan", "voxbloxpy")

    def __init__(
        self,
        host_port_pairs: List[HostPortPair],
        use_available_host: bool = False,
        force_continue: bool = False,
        n_measure_sample: int = 40,
    ):
        self.hostport_cpuinfo_map = self._init_get_cpu_infos(host_port_pairs, use_available_host)
        list(self.hostport_cpuinfo_map.keys())
        # self._init_check_dependent_module_hash(available_hostport_pairs, force_continue)
        self.n_measure_sample = n_measure_sample

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
        hostport: HostPortPair, request: MainRequestT, queue: Queue
    ) -> None:
        logger.debug("send_and_recive_and_get_elapsed_time called on pid: {}".format(os.getpid()))
        with http_connection(*hostport) as conn:
            response = send_request(conn, request)
        logger.debug("send_and_recive_and_get_elapsed_time finished on pid: {}".format(os.getpid()))
        queue.put((hostport, response.elapsed_time))

    def _measure_performance_of_each_server(
        self,
        request: MainRequestT,
    ) -> Dict[HostPortPair, float]:
        assert request.n_process == -1

        logger.info("measure performance of each server by letting them make a dummy dataset")
        score_map: Dict[HostPortPair, float] = {}
        with tempfile.TemporaryDirectory() as td:
            Path(td)
            queue = Queue()  # type: ignore
            process_list = []
            for hostport in self.hostport_cpuinfo_map.keys():
                cpu_info = self.hostport_cpuinfo_map[hostport]
                request.n_process = cpu_info.n_cpu
                p = Process(
                    target=self._send_and_recive_and_get_elapsed_time,
                    args=(hostport, request, queue),
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
