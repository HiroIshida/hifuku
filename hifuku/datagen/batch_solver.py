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
from typing import Generic, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import threadpoolctl
import tqdm
from rpbench.interface import TaskBase
from skmp.solver.interface import AbstractScratchSolver, ConfigT, ResultT
from skmp.trajectory import Trajectory

from hifuku.config import ServerSpec
from hifuku.datagen.http_datagen.client import ClientBase
from hifuku.datagen.http_datagen.request import (
    SolveProblemRequest,
    http_connection,
    send_request,
)
from hifuku.datagen.utils import split_indices
from hifuku.types import _CLAMP_FACTOR

logger = logging.getLogger(__name__)

TrajectoryMaybeList = Union[List[Trajectory], Trajectory]


def duplicate_init_solution_if_not_list(
    init_solution: Optional[TrajectoryMaybeList], n_inner_task: int
) -> Sequence[Optional[Trajectory]]:
    init_solutions: Sequence[Optional[Trajectory]]
    if isinstance(init_solution, List):
        init_solutions = init_solution
    else:
        init_solutions = [init_solution] * n_inner_task
    return init_solutions


class BatchProblemSolver(Generic[ConfigT, ResultT], ABC):
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
        task_paramss: np.ndarray,
        init_solutions: Sequence[Optional[TrajectoryMaybeList]],
        use_default_solver: bool = False,
        tmp_n_max_call_mult_factor: float = 1.0,
    ) -> List[Tuple[ResultT, ...]]:
        """
        tmp_n_max_call_mult_factor is used to increase the n_max_call temporarily.
        This is beneficiall when we want to train a iteration predictor such taht
        inference of n_call around n_max_call is accurate. Wihtout this, we will not
        have any data of n_call > n_max_call and the inference around there will be
        quite inaccurate. This technique is especially important for dataset genration
        for iteration predictor of a randomized algorithm.
        """
        # FIXME: dirty hack (A)
        # probably, making config be a function argument is cleaner

        # if mult_factor is greater than _CLAMP_FACTOR, then the infeasible problem will be
        # treated as "easier" problem than feasible problem. so...
        assert tmp_n_max_call_mult_factor <= _CLAMP_FACTOR[0]

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
        # Thus, in the following, we first measure the pickle size, and splits the problem set
        # and then send the multiple chunk of problems sequentially.
        if self.n_limit_batch is None:
            logger.debug("n_limit_batch is not set. detremine now...")
            max_ram_usage = 16 * 10**9
            task_for_measure_size = self.task_type.from_intrinsic_desc_vecs(task_paramss[0])
            serialize_ram_size_each = len(pickle.dumps(task_for_measure_size)) * 2
            max_size = int(max_ram_usage // serialize_ram_size_each)
            logger.debug(
                f"max_ram_usage: {max_ram_usage}, serialize_ram_size_each: {serialize_ram_size_each}, max_size: {max_size}"
            )
        else:
            logger.debug("use prescribed n_limit_batch {}".format(self.n_limit_batch))
            max_size = self.n_limit_batch
        logger.debug("max_size is set to {}".format(max_size))

        indices = range(len(task_paramss))
        indices_list = np.array_split(indices, np.ceil(len(task_paramss) / max_size))

        resultss = []
        for indices_part in indices_list:
            init_solutions_est_list_part = [init_solutions[i] for i in indices_part]
            results_part = self._solve_batch_impl(task_paramss[indices_part], init_solutions_est_list_part, use_default_solver=use_default_solver)  # type: ignore
            resultss.extend(results_part)

        # FIXME: dirty hack (B)
        self.config.n_max_call = n_max_call_original
        return resultss

    @abstractmethod
    def _solve_batch_impl(
        self,
        task_paramss: np.ndarray,
        init_solutions: Sequence[Optional[TrajectoryMaybeList]],
        use_default_solver: bool = False,
    ) -> List[Tuple[ResultT, ...]]:
        ...


class MultiProcessBatchProblemSolver(BatchProblemSolver[ConfigT, ResultT]):
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
        task_paramss: np.ndarray,
        init_solutions: Sequence[Optional[TrajectoryMaybeList]],
        use_default_solver: bool = False,
    ) -> List[Tuple[ResultT, ...]]:

        assert len(task_paramss) == len(init_solutions)
        assert len(task_paramss) > 0

        n_process = min(self.n_process, len(task_paramss))
        logger.info("*n_process: {}".format(n_process))
        logger.info("use_default_solver: {}".format(use_default_solver))

        is_single_process = n_process == 1
        if is_single_process:

            if use_default_solver:
                # NOTE: sovle_default does not return ResultT ...
                # Maybe we should replace ResultT with ResultProtocol ??
                # return [tuple(task.solve_default()) for task in task_params]  # type: ignore
                resultss = []
                for task_params in task_paramss:
                    task = self.task_type.from_intrinsic_desc_vecs(task_params)
                    results = tuple(task.solve_default())
                    resultss.append(results)
                return resultss
            else:
                results_list: List[Tuple[ResultT, ...]] = []
                n_max_call = self.config.n_max_call
                logger.debug("*n_max_call: {}".format(n_max_call))

                solver = self.solver_t.init(self.config)
                for task_params, init_solution in zip(task_paramss, init_solutions):
                    task = self.task_type.from_intrinsic_desc_vecs(task_params)
                    init_solutions_per_inner = duplicate_init_solution_if_not_list(
                        init_solution, task.n_inner_task
                    )
                    problems = task.export_problems()
                    results: List[ResultT] = []
                    for problem, init_solution_per_inner in zip(problems, init_solutions_per_inner):
                        solver.setup(problem)
                        result = solver.solve(init_solution_per_inner)
                        results.append(result)
                    results_list.append(tuple(results))
                return results_list
        else:
            # python's known bug when forking process while using logging module
            # https://stackoverflow.com/questions/65080123/python-multiprocessing-pool-some-process-in-deadlock-when-forked-but-runs-when-s
            # https://stackoverflow.com/questions/24509650/deadlock-with-logging-multiprocess-multithread-python-script
            # https://bugs.python.org/issue6721
            for hn in logger.handlers:
                assert hn.lock is not None
                assert not hn.lock.locked()

            args = []
            for idx in range(len(task_paramss)):
                args.append((idx, task_paramss[idx], init_solutions[idx]))

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
            _, resultss = zip(*idx_results_pairs_sorted)
            return list(resultss)

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

        task_idx, task_params, init_solutions = args
        task = _task_type.from_intrinsic_desc_vecs(task_params)  # type: ignore
        # NOTE: a lot of type: ignore due to global variables
        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            if _use_default_solver:  # type: ignore
                results = task.solve_default()
            else:
                init_solutions = duplicate_init_solution_if_not_list(
                    init_solutions, task.n_inner_task
                )
                results = []
                for problem, init_solution in zip(task.export_problems(), init_solutions):
                    _solver.setup(problem)  # type: ignore
                    result = _solver.solve(init_solution)  # type: ignore
                    results.append(result)
        return task_idx, tuple(results)


HostPortPair = Tuple[str, int]


class DistributedBatchProblemSolver(
    ClientBase[SolveProblemRequest], BatchProblemSolver[ConfigT, ResultT]
):
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
        BatchProblemSolver.__init__(self, solver_t, config, task_type, n_limit_batch)
        ClientBase.__init__(
            self, server_specs, use_available_host, force_continue, n_measure_sample
        )
        self.n_process_per_server = n_process_per_server

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

    def _solve_batch_impl(
        self,
        task_paramss: np.ndarray,
        init_solutions: Sequence[Optional[TrajectoryMaybeList]],
        use_default_solver: bool = False,
    ) -> List[Tuple[ResultT, ...]]:
        logger.debug("use_default_solver: {}".format(use_default_solver))

        hostport_pairs = list(self.hostport_cpuinfo_map.keys())
        task_paramss_measure = task_paramss[: self.n_measure_sample]
        init_solutions_measure = init_solutions[: self.n_measure_sample]
        request_for_measure = SolveProblemRequest(
            task_paramss_measure,
            self.solver_t,
            self.config,
            self.task_type,
            init_solutions_measure,
            -1,
            use_default_solver,
        )
        n_problem = len(task_paramss)
        n_problem_table = self.create_gen_number_table(request_for_measure, n_problem)
        indices_list = split_indices(n_problem, list(n_problem_table.values()))

        # send request
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            process_list = []
            for hostport, indices in zip(hostport_pairs, indices_list):
                if self.n_process_per_server is None:
                    n_process = self.hostport_cpuinfo_map[hostport].n_cpu
                else:
                    n_process = self.n_process_per_server
                task_paramss_part = task_paramss[indices]
                init_solutions_part = [init_solutions[i] for i in indices]
                req = SolveProblemRequest(
                    task_paramss_part,
                    self.solver_t,
                    self.config,
                    self.task_type,
                    init_solutions_part,
                    n_process,
                    use_default_solver,
                )
                if len(task_paramss_part) > 0:
                    p = Process(
                        target=self.send_and_recive_and_write,
                        args=(hostport, req, indices, td_path),
                    )
                    p.start()
                    process_list.append(p)

            for p in process_list:
                p.join()

            results_list_all: List[List[ResultT]] = []
            indices_all: List[int] = []
            for file_path in td_path.iterdir():
                with file_path.open(mode="rb") as f:
                    try:
                        indices_part, results_list_part = pickle.load(f)
                    except Exception as e:
                        temp_file_path = "/tmp/malignant_pickle.pkl"
                        shutil.move(str(file_path), temp_file_path)
                        logger.error(
                            "loading pickle file failed. move the file at issue to {} for the future analysis."
                        )
                        raise e

                    results_list_all.extend(results_list_part)
                    indices_all.extend(indices_part)

            idx_result_pairs = list(zip(indices_all, results_list_all))
            idx_result_pairs_sorted = sorted(idx_result_pairs, key=lambda x: x[0])  # type: ignore
            _, results = zip(*idx_result_pairs_sorted)
            ret = list(results)
            assert len(ret) == len(task_paramss)
        return ret  # type: ignore
