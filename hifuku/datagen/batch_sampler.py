import logging
import multiprocessing
import os
import pickle
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
from hifuku.datagen.utils import split_number
from hifuku.pool import PredicatedProblemPool, ProblemT
from hifuku.utils import get_random_seed, num_torch_thread

logger = logging.getLogger(__name__)

HostPortPair = Tuple[str, int]


@dataclass
class BatchProblemSampler(Generic[ProblemT], ABC):
    @abstractmethod
    def sample_batch(
        self, n_sample: int, pool: PredicatedProblemPool[ProblemT], invalidate_gridsdf: bool = False
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
        invalidate_gridsdf: bool,
        cache_path: Path,
    ) -> None:

        # set random seed
        unique_seed = get_random_seed()
        np.random.seed(unique_seed)
        logger.debug("random seed set to {}".format(unique_seed))

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
                            if invalidate_gridsdf:
                                problem.invalidate_gridsdf()
                            problems.append(problem)
                            pbar.update(1)
        ts = time.time()
        file_path = cache_path / str(uuid.uuid4())
        with file_path.open(mode="wb") as f:
            pickle.dump(problems, f)
        logger.debug("time to dump {}".format(time.time() - ts))

    def sample_batch(
        self, n_sample: int, pool: PredicatedProblemPool[ProblemT], invalidate_gridsdf: bool = False
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
                if n_sample_part == 0:
                    continue
                show_progress = idx_process == 0
                args = (
                    n_sample_part,
                    pool,
                    show_progress,
                    self.n_thread,
                    invalidate_gridsdf,
                    td_path,
                )
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
        self, n_sample: int, pool: PredicatedProblemPool[ProblemT], invalidate_gridsdf: bool = False
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
                    req = SampleProblemRequest(n_sample_part, pool, n_process, invalidate_gridsdf)
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
