import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Process, Queue, get_context
from typing import List, Optional, Sequence, Tuple

import numpy as np

from hifuku.coverage import CoverageResult, DetermineMarginsResult, determine_margins
from hifuku.datagen.http_datagen.client import ClientBase
from hifuku.datagen.http_datagen.request import (
    DetermineMarginsRequest,
    DetermineMarginsResponse,
    http_connection,
    send_request,
)
from hifuku.datagen.utils import split_number
from hifuku.utils import get_random_seed

logger = logging.getLogger(__name__)

HostPortPair = Tuple[str, int]


@dataclass
class BatchMarginDeterminant(ABC):
    @abstractmethod
    def determine_batch(
        self,
        n_sample: int,
        coverage_results: List[CoverageResult],
        threshold: float,
        target_fp_rate: float,
        cma_sigma: float,
        margins_guess: Optional[np.ndarray] = None,
        minimum_coverage: Optional[float] = None,
    ) -> Sequence[Optional[DetermineMarginsResult]]:
        ...


class MultiProcesBatchMarginDeterminant(BatchMarginDeterminant):
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
        queue: Queue,
        coverage_results: List[CoverageResult],
        threshold: float,
        target_fp_rate: float,
        cma_sigma: float,
        margins_guess: Optional[np.ndarray] = None,
        minimum_coverage: Optional[float] = None,
    ) -> None:

        # set random seed
        unique_seed = get_random_seed()
        np.random.seed(unique_seed)
        logger.debug("random seed set to {}".format(unique_seed))

        for _ in range(n_sample):
            ret = determine_margins(
                coverage_results,
                threshold,
                target_fp_rate,
                cma_sigma,
                margins_guess,
                minimum_coverage,
            )
            queue.put(ret)

    def determine_batch(
        self,
        n_sample: int,
        coverage_results: List[CoverageResult],
        threshold: float,
        target_fp_rate: float,
        cma_sigma: float,
        margins_guess: Optional[np.ndarray] = None,
        minimum_coverage: Optional[float] = None,
    ) -> Sequence[Optional[DetermineMarginsResult]]:

        ctx = get_context(method="fork")
        n_sample_list = split_number(n_sample, self.n_process)
        process_list = []
        queue: Queue[Optional[DetermineMarginsResult]] = Queue()

        for idx_process, n_sample_part in enumerate(n_sample_list):
            args = (
                n_sample_part,
                queue,
                coverage_results,
                threshold,
                target_fp_rate,
                cma_sigma,
                margins_guess,
                minimum_coverage,
            )
            p = ctx.Process(target=self.work, args=args)  # type: ignore
            p.start()
            process_list.append(p)

        # the result is margins, coverage, fprate order
        result_list: List[Optional[DetermineMarginsResult]] = [queue.get() for _ in range(n_sample)]
        for p in process_list:
            p.join()
        return result_list


class DistributeBatchMarginDeterminant(ClientBase[DetermineMarginsRequest], BatchMarginDeterminant):
    @staticmethod  # called only in generate
    def send_and_recive_and_put(
        hostport: HostPortPair, request: DetermineMarginsRequest, queue: Queue
    ) -> None:
        logger.debug("send_and_recive_and_put called on pid: {}".format(os.getpid()))
        with http_connection(*hostport) as conn:
            response: DetermineMarginsResponse = send_request(conn, request)  # type: ignore
        for result in response.results:
            queue.put(result)
        logger.debug("send_and_recive_and_put finished on pid: {}".format(os.getpid()))

    def determine_batch(
        self,
        n_sample: int,
        coverage_results: List[CoverageResult],
        threshold: float,
        target_fp_rate: float,
        cma_sigma: float,
        margins_guess: Optional[np.ndarray] = None,
        minimum_coverage: Optional[float] = None,
    ) -> Sequence[Optional[DetermineMarginsResult]]:

        hostport_pairs = list(self.hostport_cpuinfo_map.keys())
        n_sample_table = self.create_gen_number_table(None, n_sample)

        queue: Queue[Optional[DetermineMarginsResult]] = Queue()
        process_list = []
        for hostport in hostport_pairs:
            n_sample_part = n_sample_table[hostport]
            if n_sample_part > 0:
                n_process = self.hostport_cpuinfo_map[hostport].n_cpu
                req = DetermineMarginsRequest(
                    n_sample_part,
                    n_process,
                    coverage_results,
                    threshold,
                    target_fp_rate,
                    cma_sigma,
                    margins_guess,
                    minimum_coverage,
                )

                p = Process(target=self.send_and_recive_and_put, args=(hostport, req, queue))
                p.start()
                process_list.append(p)

        # the result is margins, coverage, fprate order
        result_list: List[Optional[DetermineMarginsResult]] = [queue.get() for _ in range(n_sample)]

        for p in process_list:
            p.join()
        return result_list
