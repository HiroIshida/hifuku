import logging
import multiprocessing
import os
import pickle
import tempfile
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Generic, List, Optional, Tuple

import numpy as np
import threadpoolctl
import tqdm

from hifuku.datagen.http_datagen.client import ClientBase
from hifuku.datagen.http_datagen.request import (
    SampleProblemRequest,
    http_connection,
    send_request,
)
from hifuku.pool import PredicatedProblemPool, ProblemT
from hifuku.utils import determine_process_thread, get_random_seed, num_torch_thread

logger = logging.getLogger(__name__)

HostPortPair = Tuple[str, int]


@dataclass
class BatchProblemSampler(Generic[ProblemT], ABC):
    @abstractmethod
    def sample_batch(
        self, n_sample: int, pool: PredicatedProblemPool[ProblemT], delete_cache: bool = False
    ) -> List[ProblemT]:
        ...


class MultiProcessBatchProblemSampler(BatchProblemSampler[ProblemT]):
    n_process: int
    n_thread: int

    def __init__(self, n_process: Optional[int] = None):
        n_process_default, n_thread = determine_process_thread()
        if n_process is None:
            n_process = n_process_default
        logger.info("n_process: {}, n_thread: {}".format(n_process, n_thread))
        self.n_process = n_process
        self.n_thread = n_thread

    def sample_batch(
        self, n_sample: int, pool: PredicatedProblemPool[ProblemT], delete_cache: bool = False
    ) -> List[ProblemT]:
        assert pool.parallelizable()
        assert n_sample > 0
        n_process = min(self.n_process, n_sample)

        with ProcessPoolExecutor(
            n_process,
            initializer=self._process_pool_setup,
            initargs=(pool, self.n_thread, delete_cache),
            mp_context=get_context("fork"),
        ) as executor:
            problems_sampled = list(
                tqdm.tqdm(
                    executor.map(self._process_pool_sample_task, range(n_sample)), total=n_sample
                )
            )
        return problems_sampled

    @staticmethod
    def _process_pool_setup(
        _pool: PredicatedProblemPool[ProblemT], _n_thread: int, _delete_cache: bool
    ) -> None:
        global pool, n_thread, delete_cache  # shared in the forked process
        pool = _pool  # type: ignore
        n_thread = _n_thread  # type: ignore
        delete_cache = _delete_cache  # type: ignore

        unique_seed = get_random_seed()
        np.random.seed(unique_seed)
        logger.debug("random seed set to {}".format(unique_seed))

    @staticmethod
    def _process_pool_sample_task(_) -> ProblemT:
        global pool, n_thread, delete_cache  # shared in the forked process

        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            with num_torch_thread(n_thread):  # type: ignore
                while True:
                    task = next(pool)  # type: ignore
                    if task is not None:
                        if delete_cache:  # type: ignore
                            task.delete_cache()
                        return task


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
        self, n_sample: int, pool: PredicatedProblemPool[ProblemT], delete_cache: bool = False
    ) -> List[ProblemT]:
        assert n_sample > 0
        assert pool.parallelizable()

        hostport_pairs = list(self.hostport_cpuinfo_map.keys())
        request_for_measure = SampleProblemRequest(self.n_measure_sample, pool, -1, True)
        n_sample_table = self.create_gen_number_table(request_for_measure, n_sample)

        # NOTE: after commit 18c664f, process starts hang with forking.
        # spawn is slower than fork, but number of spawining processes is at most several
        # in this case, so it's almost no cost
        ctx = multiprocessing.get_context(method="spawn")
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            process_list = []
            for hostport in hostport_pairs:
                n_sample_part = n_sample_table[hostport]
                if n_sample_part > 0:
                    n_process = self.hostport_cpuinfo_map[hostport].n_cpu
                    req = SampleProblemRequest(n_sample_part, pool, n_process, delete_cache)
                    p = ctx.Process(
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
