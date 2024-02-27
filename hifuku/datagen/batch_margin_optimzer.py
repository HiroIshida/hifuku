import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Process, Queue, get_context
from typing import List, Optional, Sequence, Tuple

import numpy as np
import threadpoolctl

from hifuku.coverage import OptimizeMarginsResult, RealEstAggregate, optimize_margins
from hifuku.datagen.http_datagen.client import ClientBase
from hifuku.datagen.http_datagen.request import (
    OptimizeMarginsRequest,
    OptimizeMarginsResponse,
    http_connection,
    send_request,
)
from hifuku.datagen.utils import split_number
from hifuku.utils import determine_process_thread, get_random_seed

logger = logging.getLogger(__name__)

HostPortPair = Tuple[str, int]


@dataclass
class BatchMarginsOptimizerBase(ABC):
    @abstractmethod
    def optimize_batch(
        self,
        n_sample: int,
        aggregate_list: List[RealEstAggregate],
        threshold: float,
        target_fp_rate: float,
        cma_sigma: float,
        margins_guess: Optional[np.ndarray] = None,
        minimum_coverage: Optional[float] = None,
    ) -> Sequence[Optional[OptimizeMarginsResult]]:
        ...


class MultiProcesBatchMarginsOptimizer(BatchMarginsOptimizerBase):
    n_process: int
    n_thread: int

    def __init__(self, n_process: Optional[int] = None):
        n_process_default, n_thread = determine_process_thread()
        if n_process is None:
            n_process = n_process_default
        logger.info("n_process: {}, n_thread: {}".format(n_process, n_thread))
        self.n_process = n_process
        self.n_thread = n_thread

    @staticmethod
    def work(
        n_sample: int,
        queue: Queue,
        aggregate_list: List[RealEstAggregate],
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

        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            for _ in range(n_sample):
                ret = optimize_margins(
                    aggregate_list,
                    threshold,
                    target_fp_rate,
                    cma_sigma,
                    margins_guess,
                    minimum_coverage,
                )
                queue.put(ret)

    def optimize_batch(
        self,
        n_sample: int,
        aggregate_list: List[RealEstAggregate],
        threshold: float,
        target_fp_rate: float,
        cma_sigma: float,
        margins_guess: Optional[np.ndarray] = None,
        minimum_coverage: Optional[float] = None,
    ) -> Sequence[Optional[OptimizeMarginsResult]]:

        ctx = get_context(method="fork")
        n_sample_list = split_number(n_sample, self.n_process)
        process_list = []
        queue: Queue[Optional[OptimizeMarginsResult]] = Queue()

        for idx_process, n_sample_part in enumerate(n_sample_list):
            args = (
                n_sample_part,
                queue,
                aggregate_list,
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
        result_list: List[Optional[OptimizeMarginsResult]] = [queue.get() for _ in range(n_sample)]
        for p in process_list:
            p.join()
        return result_list


class DistributeBatchMarginsOptimizer(
    ClientBase[OptimizeMarginsRequest], BatchMarginsOptimizerBase
):
    @staticmethod  # called only in generate
    def send_and_recive_and_put(
        hostport: HostPortPair, request: OptimizeMarginsRequest, queue: Queue
    ) -> None:
        logger.debug("send_and_recive_and_put called on pid: {}".format(os.getpid()))
        with http_connection(*hostport) as conn:
            response: OptimizeMarginsResponse = send_request(conn, request)  # type: ignore
        for result in response.results:
            queue.put(result)
        logger.debug("send_and_recive_and_put finished on pid: {}".format(os.getpid()))

    def optimize_batch(
        self,
        n_sample: int,
        aggregate_list: List[RealEstAggregate],
        threshold: float,
        target_fp_rate: float,
        cma_sigma: float,
        margins_guess: Optional[np.ndarray] = None,
        minimum_coverage: Optional[float] = None,
    ) -> Sequence[Optional[OptimizeMarginsResult]]:

        hostport_pairs = list(self.hostport_cpuinfo_map.keys())
        n_sample_table = self.determine_assignment_per_server(n_sample)

        queue: Queue[Optional[OptimizeMarginsResult]] = Queue()
        process_list = []
        for hostport in hostport_pairs:
            n_sample_part = n_sample_table[hostport]
            if n_sample_part > 0:
                n_process = self.hostport_cpuinfo_map[hostport].n_cpu
                req = OptimizeMarginsRequest(
                    n_sample_part,
                    n_process,
                    aggregate_list,
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
        result_list: List[Optional[OptimizeMarginsResult]] = [queue.get() for _ in range(n_sample)]

        for p in process_list:
            p.join()
        return result_list
