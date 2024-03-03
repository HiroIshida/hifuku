import logging
import os
import pickle
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, get_context
from pathlib import Path
from typing import Generic, List, Optional, Tuple, Type, Union

import numpy as np
import threadpoolctl
import tqdm
from rpbench.interface import TaskBase
from skmp.solver.interface import AbstractScratchSolver, ConfigT, ResultT
from skmp.trajectory import Trajectory

from hifuku.datagen.http_datagen.client import ClientBase, ServerSpec
from hifuku.datagen.http_datagen.request import (
    SolveTaskRequest,
    http_connection,
    send_request,
)
from hifuku.datagen.utils import split_indices

logger = logging.getLogger(__name__)

TrajectoryMaybeList = Union[List[Trajectory], Trajectory]


class BatchTaskSolver(Generic[ConfigT, ResultT], ABC):
    solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]]
    task_type: Type[TaskBase]
    config: ConfigT
    n_limit_batch: Optional[int]

    def __init__(
        self,
        solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]],
        config: ConfigT,
        task_type: Type[TaskBase],
        n_limit_batch: Optional[int] = None,
    ):
        self.solver_t = solver_t
        self.config = config
        self.task_type = task_type
        self.n_limit_batch = n_limit_batch

    def solve_batch(
        self,
        task_params: np.ndarray,
        init_solutions: Optional[List[Trajectory]] = None,
        use_default_solver: bool = False,
        tmp_n_max_call_mult_factor: float = 1.0,
    ) -> List[ResultT]:
        """
        tmp_n_max_call_mult_factor is used to increase the n_max_call temporarily.
        This is beneficiall when we want to train a cost predictor such taht
        inference of n_call around n_max_call is accurate. Wihtout this, we will not
        have any data of n_call > n_max_call and the inference around there will be
        quite inaccurate. This technique is especially important for dataset genration
        for cost predictor of a randomized algorithm.
        """
        # FIXME: dirty hack (A)
        # probably, making config be a function argument is cleaner

        n_max_call_original = self.config.n_max_call
        self.config.n_max_call = int(n_max_call_original * tmp_n_max_call_mult_factor)
        logger.debug(
            "temporarily increase n_max_call from {} to {}".format(
                n_max_call_original, self.config.n_max_call
            )
        )

        # When pickling-and-depickling, the procedure takes up much more memory than
        # pickled object size. I'm not sure but according to https://stackoverflow.com/a/38971446/7624196
        # the RAM usage could be two times bigger than the serialized-object size.
        # Thus, in the following, we first measure the pickle size, and splits the task set
        # and then send the multiple chunk of tasks sequentially.
        if self.n_limit_batch is None:
            logger.debug("n_limit_batch is not set. detremine now...")
            max_ram_usage = 16 * 10**9
            task_for_measure_size = self.task_type.from_task_param(task_params[0])
            serialize_ram_size_each = len(pickle.dumps(task_for_measure_size)) * 2
            max_size = int(max_ram_usage // serialize_ram_size_each)
            logger.debug(
                f"max_ram_usage: {max_ram_usage}, serialize_ram_size_each: {serialize_ram_size_each}, max_size: {max_size}"
            )
        else:
            logger.debug("use prescribed n_limit_batch {}".format(self.n_limit_batch))
            max_size = self.n_limit_batch
        logger.debug("max_size is set to {}".format(max_size))

        indices = range(len(task_params))
        indices_list = np.array_split(indices, np.ceil(len(task_params) / max_size))

        if init_solutions is None:
            init_solutions = [None] * len(task_params)

        results = []
        for indices_part in indices_list:
            init_solutions_est_list_part = [init_solutions[i] for i in indices_part]
            results_part = self._solve_batch_impl(task_params[indices_part], init_solutions_est_list_part, use_default_solver=use_default_solver)  # type: ignore
            results.extend(results_part)

        # FIXME: dirty hack (B)
        self.config.n_max_call = n_max_call_original
        return results

    @abstractmethod
    def _solve_batch_impl(
        self,
        task_params: np.ndarray,
        init_solutions: Union[List[Trajectory], List[None]],
        use_default_solver: bool = False,
    ) -> List[ResultT]:
        ...


class MultiProcessBatchTaskSolver(BatchTaskSolver[ConfigT, ResultT]):
    n_process: int

    def __init__(
        self,
        solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]],
        config: ConfigT,
        task_type: Type[TaskBase],
        n_process: Optional[int] = None,
        n_limit_batch: Optional[int] = None,
    ):
        super().__init__(solver_t, config, task_type, n_limit_batch)
        if n_process is None:
            logger.info("n_process is not set. automatically determine")
            cpu_num = os.cpu_count()
            assert cpu_num is not None
            n_process = int(cpu_num * 0.5)
        logger.info("n_process is set to {}".format(n_process))
        self.n_process = n_process

    def _solve_batch_impl(
        self,
        task_params: np.ndarray,
        init_solutions: Union[List[Trajectory], List[None]],
        use_default_solver: bool = False,
    ) -> List[ResultT]:

        assert len(task_params) == len(init_solutions)
        assert len(task_params) > 0

        n_process = min(self.n_process, len(task_params))
        logger.info("*n_process: {}".format(n_process))
        logger.info("use_default_solver: {}".format(use_default_solver))

        is_single_process = n_process == 1
        if is_single_process:

            if use_default_solver:
                # NOTE: sovle_default does not return ResultT ...
                # Maybe we should replace ResultT with ResultProtocol ??
                # return [tuple(task.solve_default()) for task in task_params]  # type: ignore
                results = []
                for task_param in task_params:
                    task = self.task_type.from_task_param(task_param)
                    result = task.solve_default()
                    results.append(result)
                return results
            else:
                n_max_call = self.config.n_max_call
                logger.debug("*n_max_call: {}".format(n_max_call))

                solver = self.solver_t.init(self.config)
                results = []
                for task_param, init_solution in zip(task_params, init_solutions):
                    assert not isinstance(init_solution, list)
                    task = self.task_type.from_task_param(task_param)
                    problem = task.export_problem()
                    solver.setup(problem)
                    result = solver.solve(init_solution)
                    results.append(result)
                return results
        else:
            # python's known bug when forking process while using logging module
            # https://stackoverflow.com/questions/65080123/python-multiprocessing-pool-some-process-in-deadlock-when-forked-but-runs-when-s
            # https://stackoverflow.com/questions/24509650/deadlock-with-logging-multiprocess-multithread-python-script
            # https://bugs.python.org/issue6721
            for hn in logger.handlers:
                assert hn.lock is not None
                assert not hn.lock.locked()

            args = []
            for idx in range(len(task_params)):
                args.append((idx, task_params[idx], init_solutions[idx]))

            with ProcessPoolExecutor(
                n_process,
                initializer=self._pool_setup,
                initargs=(self.solver_t, self.config, self.task_type, use_default_solver),
                mp_context=get_context("fork"),
            ) as executor:
                idx_results_pairs = list(
                    tqdm.tqdm(executor.map(self._pool_solve_single, args), total=len(args))
                )
            idx_results_pairs_sorted = sorted(idx_results_pairs, key=lambda x: x[0])  # type: ignore
            _, results = zip(*idx_results_pairs_sorted)
            return results

    @staticmethod
    def _pool_setup(  # used only in process pool
        solver_t: Type[AbstractScratchSolver],
        config: ConfigT,
        task_type: Type[TaskBase],
        use_default_solver: bool,
    ):
        from hifuku.script_utils import filter_warnings

        filter_warnings()
        global _solver
        if not use_default_solver:
            _solver = solver_t.init(config)  # type: ignore
        global _use_default_solver
        _use_default_solver = use_default_solver  # type: ignore
        global _task_type
        _task_type = task_type  # type: ignore

    @staticmethod
    def _pool_solve_single(args):  # used only in process pool
        global _solver
        global _use_default_solver
        global _task_type

        task_idx, task_param, init_solution = args
        task = _task_type.from_task_param(task_param)  # type: ignore
        # NOTE: a lot of type: ignore due to global variables
        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            if _use_default_solver:  # type: ignore
                result = task.solve_default()
            else:
                problem = task.export_problem()
                _solver.setup(problem)
                result = _solver.solve(init_solution)
        return task_idx, result


HostPortPair = Tuple[str, int]


class DistributedBatchTaskSolver(ClientBase[SolveTaskRequest], BatchTaskSolver[ConfigT, ResultT]):
    n_process_per_server: Optional[int]

    def __init__(
        self,
        solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]],
        config: ConfigT,
        task_type: Type[TaskBase],
        server_specs: Optional[Tuple[ServerSpec, ...]] = None,
        use_available_host: bool = False,
        force_continue: bool = False,
        n_measure_sample: int = 40,
        n_process_per_server: Optional[int] = None,
        n_limit_batch: Optional[int] = None,
    ):
        BatchTaskSolver.__init__(self, solver_t, config, task_type, n_limit_batch)
        ClientBase.__init__(
            self, server_specs, use_available_host, force_continue, n_measure_sample
        )
        self.n_process_per_server = n_process_per_server

    @staticmethod  # called only in generate
    def send_and_recive_and_write(
        hostport: HostPortPair, request: SolveTaskRequest, indices: np.ndarray, tmp_path: Path
    ) -> None:

        logger.debug("send_and_recive_and_write called on pid: {}".format(os.getpid()))
        with http_connection(*hostport) as conn:
            response = send_request(conn, request)
        file_path = tmp_path / str(uuid.uuid4())
        with file_path.open(mode="wb") as f:
            pickle.dump((indices, response.results), f)
        logger.debug("saved to {}".format(file_path))
        logger.debug("send_and_recive_and_write finished on pid: {}".format(os.getpid()))

    def _solve_batch_impl(
        self,
        task_params: np.ndarray,
        init_solutions: Union[List[Trajectory], List[None]],
        use_default_solver: bool = False,
    ) -> List[ResultT]:
        try:
            return self._solve_batch_impl_inner(
                task_params, init_solutions, use_default_solver=use_default_solver
            )
        except ValueError:
            logger.error("Probably something wrong with connection. Retry...")
            return self._solve_batch_impl_inner(
                task_params, init_solutions, use_default_solver=use_default_solver
            )

    def _solve_batch_impl_inner(
        self,
        task_params: np.ndarray,
        init_solutions: Union[List[Trajectory], List[None]],
        use_default_solver: bool = False,
    ) -> List[ResultT]:
        logger.debug("use_default_solver: {}".format(use_default_solver))

        n_task = len(task_params)
        n_task_table = self.determine_assignment_per_server(n_task)
        indices_list = split_indices(n_task, list(n_task_table.values()))

        # send request
        hostport_pairs = list(self.hostport_cpuinfo_map.keys())
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            process_list = []
            for hostport, indices in zip(hostport_pairs, indices_list):
                if self.n_process_per_server is None:
                    n_process = self.hostport_cpuinfo_map[hostport].n_cpu
                else:
                    n_process = self.n_process_per_server
                task_params_part = task_params[indices]
                init_solutions_part = [init_solutions[i] for i in indices]
                req = SolveTaskRequest(
                    task_params_part,
                    self.solver_t,
                    self.config,
                    self.task_type,
                    init_solutions_part,
                    n_process,
                    use_default_solver,
                )
                if len(task_params_part) > 0:
                    p = Process(
                        target=self.send_and_recive_and_write,
                        args=(hostport, req, indices, td_path),
                    )
                    p.start()
                    process_list.append(p)

            for p in process_list:
                p.join()

            results_all: List[ResultT] = []
            indices_all: List[int] = []
            for file_path in td_path.iterdir():
                with file_path.open(mode="rb") as f:
                    try:
                        indices_part, results_part = pickle.load(f)
                    except Exception as e:
                        temp_file_path = "/tmp/malignant_pickle.pkl"
                        shutil.move(str(file_path), temp_file_path)
                        logger.error(
                            "loading pickle file failed. move the file at issue to {} for the future analysis."
                        )
                        raise e

                    results_all.extend(results_part)
                    indices_all.extend(indices_part)

            idx_result_pairs = list(zip(indices_all, results_all))
            idx_result_pairs_sorted = sorted(idx_result_pairs, key=lambda x: x[0])  # type: ignore
            _, results = zip(*idx_result_pairs_sorted)
            ret = list(results)
            if len(ret) != len(task_params):
                message = f"len(ret) != len(task_params) ({len(ret)} != {len(task_params)})"
                logger.error(message)
                raise ValueError(message)
        return ret  # type: ignore
