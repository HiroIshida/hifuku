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
    SampleTaskRequest,
    SampleTaskResponse,
    http_connection,
    send_request,
)
from hifuku.pool import PredicatedTaskPool, TaskT
from hifuku.utils import determine_process_thread, get_random_seed, num_torch_thread

logger = logging.getLogger(__name__)

HostPortPair = Tuple[str, int]


@dataclass
class BatchTaskSampler(Generic[TaskT], ABC):
    @abstractmethod
    def sample_batch(
        self, n_sample: int, pool: PredicatedTaskPool[TaskT]
    ) -> Tuple[np.ndarray, int]:
        ...


class MultiProcessBatchTaskSampler(BatchTaskSampler[TaskT]):
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
        self, n_sample: int, pool: PredicatedTaskPool[TaskT]
    ) -> Tuple[np.ndarray, int]:
        assert n_sample > 0
        n_process = min(self.n_process, n_sample)

        with ProcessPoolExecutor(
            n_process,
            initializer=self._process_pool_setup,
            initargs=(pool, self.n_thread),
            mp_context=get_context("fork"),
        ) as executor:
            # left is task_params, right is n_trial
            tmp: List[Tuple[TaskT, int]] = list(
                tqdm.tqdm(
                    executor.map(self._process_pool_sample_task, range(n_sample)), total=n_sample
                )
            )
        params = np.array([t[0] for t in tmp])
        n_trial_sum = sum(t[1] for t in tmp)

        assert params.ndim == 2
        assert params.shape[0] == n_sample
        return params, n_trial_sum

    @staticmethod
    def _process_pool_setup(_pool: PredicatedTaskPool[TaskT], _n_thread: int):
        global pool, n_thread  # shared in the forked process
        pool = _pool  # type: ignore
        n_thread = _n_thread  # type: ignore

        unique_seed = get_random_seed()
        np.random.seed(unique_seed)
        logger.debug("random seed set to {}".format(unique_seed))

    @staticmethod
    def _process_pool_sample_task(_) -> Tuple[np.ndarray, int]:
        global pool, n_thread  # shared in the forked process

        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            with num_torch_thread(n_thread):  # type: ignore
                n_trial = 0
                while True:
                    n_trial += 1
                    task = next(pool)  # type: ignore
                    if task is not None:
                        return task, n_trial


class DistributeBatchTaskSampler(ClientBase[SampleTaskRequest], BatchTaskSampler[TaskT]):
    @staticmethod  # called only in generate
    def send_and_recive_and_write(
        hostport: HostPortPair, request: SampleTaskRequest, tmp_path: Path
    ) -> None:
        logger.debug("send_and_recive_and_write called on pid: {}".format(os.getpid()))
        with http_connection(*hostport) as conn:
            response = send_request(conn, request)
        file_path = tmp_path / str(uuid.uuid4())
        assert len(response.task_params) > 0
        with file_path.open(mode="wb") as f:
            pickle.dump(response, f)
        logger.debug("saved to {}".format(file_path))
        logger.debug("send_and_recive_and_write finished on pid: {}".format(os.getpid()))

    def sample_batch(
        self, n_sample: int, pool: PredicatedTaskPool[TaskT]
    ) -> Tuple[np.ndarray, int]:
        assert n_sample > 0
        n_sample_table = self.determine_assignment_per_server(n_sample)

        # NOTE: after commit 18c664f, process starts hang with forking.
        # spawn is slower than fork, but number of spawining processes is at most several
        # in this case, so it's almost no cost
        hostport_pairs = list(self.hostport_cpuinfo_map.keys())
        ctx = multiprocessing.get_context(method="spawn")
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            process_list = []
            for hostport in hostport_pairs:
                n_sample_part = n_sample_table[hostport]
                if n_sample_part > 0:
                    n_process = self.hostport_cpuinfo_map[hostport].n_cpu
                    req = SampleTaskRequest(n_sample_part, pool, n_process)
                    p = ctx.Process(
                        target=self.send_and_recive_and_write, args=(hostport, req, td_path)
                    )
                    p.start()
                    process_list.append(p)

            for p in process_list:
                p.join()

            n_trial_total = 0
            param_list: List[np.ndarray] = []
            for file_path in td_path.iterdir():
                with file_path.open(mode="rb") as f:
                    resp: SampleTaskResponse = pickle.load(f)
                    param_list.extend(resp.task_params)
                    n_trial_total += resp.n_trial_total
        params = np.array(param_list)
        assert params.ndim == 2
        assert params.shape[0] == n_sample
        return params, n_trial_total
