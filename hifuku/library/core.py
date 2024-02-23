import copy
import datetime
import inspect
import json
import logging
import pickle
import random
import re
import shutil
import signal
import time
import uuid
from abc import abstractmethod
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Generic, List, Optional, Tuple, Type, Union

import numpy as np
import threadpoolctl
import torch
import tqdm
from mohou.trainer import TrainCache, TrainConfig, train
from ompl import set_ompl_random_seed
from rpbench.interface import AbstractTaskSolver
from skmp.solver.interface import (
    AbstractScratchSolver,
    ConfigT,
    ResultProtocol,
    ResultT,
)
from skmp.trajectory import Trajectory

from hifuku.coverage import CoverageResult
from hifuku.datagen import (
    BatchMarginsDeterminant,
    BatchProblemSampler,
    BatchProblemSolver,
    DistributeBatchMarginsDeterminant,
    DistributeBatchProblemSampler,
    DistributedBatchProblemSolver,
    MultiProcesBatchMarginsDeterminant,
    MultiProcessBatchProblemSampler,
    MultiProcessBatchProblemSolver,
)
from hifuku.neuralnet import (
    AutoEncoderBase,
    IterationPredictor,
    IterationPredictorConfig,
    IterationPredictorDataset,
    IterationPredictorWithEncoder,
    IterationPredictorWithEncoderConfig,
    NullAutoEncoder,
)
from hifuku.pool import ProblemPool, ProblemT, TrivialProblemPool
from hifuku.types import get_clamped_iter
from hifuku.utils import num_torch_thread

logger = logging.getLogger(__name__)


@dataclass
class ProfileInfo:  # per each iteration
    t_total: Optional[float] = None
    t_determine_cand: Optional[float] = None
    t_dataset: Optional[float] = None
    t_train: Optional[float] = None
    t_margin: Optional[float] = None

    @property
    def has_all_info(self) -> bool:
        return all(
            [
                self.t_total is not None,
                self.t_determine_cand is not None,
                self.t_dataset is not None,
                self.t_train is not None,
                self.t_margin is not None,
            ]
        )

    @property
    def is_valid(self) -> bool:
        if not self.has_all_info:
            return False
        # sum of each time must be smaller than t_total
        return self.t_total > (self.t_dataset + self.t_train + self.t_determine_cand + self.t_margin)  # type: ignore[operator]

    @classmethod
    def from_total(cls, t_total: float) -> "ProfileInfo":
        # for backward compatibility
        return cls(t_total, None, None, None, None)

    @property
    def t_other(self) -> float:
        return self.t_total - (self.t_dataset + self.t_train + self.t_determine_cand + self.t_margin)  # type: ignore[operator]


class ActiveSamplerState:
    # states
    sampling_number_factor: float

    # the below are not states but history for postmortem analysis
    margins_history: List[List[float]]
    coverage_results: List[CoverageResult]
    candidates_history: List[List[Trajectory]]
    elapsed_time_history: List[ProfileInfo]
    coverage_est_history: List[float]
    failure_count: int

    def __init__(self, sampling_number_factor: float):
        self.sampling_number_factor = sampling_number_factor
        self.coverage_results = []
        self.margins_history = []
        self.candidates_history = []
        self.elapsed_time_history = []
        self.coverage_est_history = []
        self.failure_count = 0

    def check_consistency(self) -> None:
        if len(self.elapsed_time_history) == 0:
            return
        assert len(self.coverage_results) == len(self.margins_history)
        total_iter = len(self.coverage_results) + self.failure_count
        assert len(self.candidates_history) == total_iter
        assert len(self.elapsed_time_history) == total_iter
        for elapsed_time in self.elapsed_time_history:
            assert elapsed_time.is_valid
        assert len(self.coverage_est_history) == total_iter
        assert all(
            [
                self.coverage_est_history[i] <= self.coverage_est_history[i + 1]
                for i in range(len(self.coverage_est_history) - 1)
            ]
        )

    def dump(self, base_path: Path) -> None:
        with (base_path / "state.pkl").open(mode="wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, base_path: Path) -> "ActiveSamplerState":
        with (base_path / "state.pkl").open(mode="rb") as f:
            state: ActiveSamplerState = pickle.load(f)
        state.check_consistency()
        return state


@dataclass
class SolutionLibrary(Generic[ProblemT, ConfigT, ResultT]):
    """Solution Library

    limitting threadnumber takes nonnegligible time. So please set
    limit_thread = false in performance evaluation time. However,
    when in attempt to run in muliple process, one must set it True.
    """

    task_type: Type[ProblemT]
    task_distribution_hash: str
    solver_type: Type[AbstractScratchSolver[ConfigT, ResultT]]
    solver_config: ConfigT
    ae_model_shared: Optional[
        AutoEncoderBase
    ]  # if None, when each predictor does not share autoencoder
    predictors: List[Union[IterationPredictor, IterationPredictorWithEncoder]]
    init_solutions: List[Trajectory]
    margins: List[float]
    uuidval: str
    meta_data: Dict
    limit_thread: bool = False

    def __post_init__(self):
        if self.ae_model_shared is not None:
            assert self.ae_model_shared.trained
            for pred in self.predictors:
                assert isinstance(pred, IterationPredictor)
        else:
            for pred in self.predictors:
                assert isinstance(pred, IterationPredictorWithEncoder)

    @dataclass
    class InferenceResult:
        nit: float
        idx: int  # index of selected solution in the library
        init_solution: Trajectory

    @classmethod
    def initialize(
        cls,
        task_type: Type[ProblemT],
        solver_type: Type[AbstractScratchSolver[ConfigT, ResultT]],
        config,
        ae_model: Optional[AutoEncoderBase],
        meta_data: Optional[Dict] = None,
    ) -> "SolutionLibrary[ProblemT, ConfigT, ResultT]":
        uuidval = str(uuid.uuid4())[-8:]
        if meta_data is None:
            meta_data = {}
        return cls(
            task_type,
            task_type.compute_distribution_hash(),
            solver_type,
            config,
            ae_model,
            [],
            [],
            [],
            uuidval,
            meta_data,
        )

    def put_on_device(self, device: torch.device):
        if self.ae_model_shared is not None:
            self.ae_model_shared.put_on_device(device)
        for pred in self.predictors:
            pred.put_on_device(device)

    @property
    def device(self) -> torch.device:
        if self.ae_model_shared is not None:
            return self.ae_model_shared.get_device()
        else:
            pred: IterationPredictorWithEncoder = self.predictors[0]  # type: ignore[assignment]
            return pred.device

    def _infer_iteration_num(self, task: ProblemT) -> np.ndarray:
        assert len(self.predictors) > 0
        has_shared_ae = self.ae_model_shared is not None
        if has_shared_ae:
            return self._infer_iteration_num_with_shared_ae(task)
        else:
            return self._infer_iteration_num_combined(task)

    def _infer_iteration_num_combined(self, task: ProblemT) -> np.ndarray:
        if self.limit_thread:
            with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                with num_torch_thread(1):
                    desc_table = task.export_table()
                    mesh_np = desc_table.get_mesh()
                    assert mesh_np is not None
                    mesh_np = np.expand_dims(mesh_np, axis=(0, 1))
                    descs_np = np.array(desc_table.get_vector_descs())
                    mesh_torch = torch.from_numpy(mesh_np).float().to(self.device)
                    descs_torch = torch.from_numpy(descs_np).float().to(self.device)
        else:
            desc_table = task.export_table()
            mesh_np = desc_table.get_mesh()
            assert mesh_np is not None
            mesh_np = np.expand_dims(mesh_np, axis=(0, 1))
            descs_np = np.array(desc_table.get_vector_descs())
            mesh_torch = torch.from_numpy(mesh_np).float().to(self.device)
            descs_torch = torch.from_numpy(descs_np).float().to(self.device)

        # these lines copied from _infer_iteration_num_with_shared_ae
        itervals_list = []
        for pred, margin in zip(self.predictors, self.margins):
            assert isinstance(pred, IterationPredictorWithEncoder)
            # margin is for correcting the overestimated inference
            itervals = pred.forward_multi_inner(mesh_torch, descs_torch)  # type: ignore
            itervals = itervals.squeeze(dim=1)
            itervals_np = itervals.detach().cpu().numpy() + margin
            itervals_list.append(itervals_np)
        itervals_arr = np.array(itervals_list)
        return itervals_arr

    def _infer_iteration_num_with_shared_ae(self, task: ProblemT) -> np.ndarray:
        """
        itervals_arr: R^{n_solution, n_desc_inner}
        """
        assert self.ae_model_shared is not None

        if self.limit_thread:
            # FIXME: maybe threadpool_limits and num_torch_thread scope can be
            # integrated into one? In that case, if-else conditioning due to
            # mesh_np_tmp can be simplar

            with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                # limiting numpy thread seems to make stable. but not sure why..
                desc_table = task.export_table()
                mesh_np_tmp = desc_table.get_mesh()
                if mesh_np_tmp is None:
                    mesh_np = None
                else:
                    mesh_np = np.expand_dims(mesh_np_tmp, axis=(0, 1))
                desc_np = np.array(desc_table.get_vector_descs())

            with num_torch_thread(1):
                # float() must be run in single (cpp-layer) thread
                # see https://github.com/pytorch/pytorch/issues/89693
                if mesh_np is None:
                    mesh = torch.empty((1, 0))
                else:
                    mesh = torch.from_numpy(mesh_np)
                    mesh = mesh.float().to(self.device)
                desc = torch.from_numpy(desc_np)
                desc = desc.float().to(self.device)
        else:
            # usually, calling threadpoolctl and num_torch_thread function
            # is constly. So if you are sure that you are running program in
            # a single process. Then set limit_thread = False
            desc_table = task.export_table()
            desc_np = np.array(desc_table.get_vector_descs())
            desc = torch.from_numpy(desc_np)
            desc = desc.float().to(self.device)

            mesh_np_tmp = desc_table.get_mesh()
            if mesh_np_tmp is None:
                mesh = torch.empty((1, 0))
            else:
                mesh_np = np.expand_dims(mesh_np_tmp, axis=(0, 1))
                mesh = torch.from_numpy(mesh_np)
                mesh = mesh.float().to(self.device)

        n_batch, _ = desc_np.shape

        encoded: torch.Tensor = self.ae_model_shared.encode(mesh)
        encoded_repeated = encoded.repeat(n_batch, 1)

        itervals_list = []
        for pred, margin in zip(self.predictors, self.margins):
            # margin is for correcting the overestimated inference
            itervals, _ = pred.forward((encoded_repeated, desc))
            itervals = itervals.squeeze(dim=1)
            itervals_np = itervals.detach().cpu().numpy() + margin
            itervals_list.append(itervals_np)
        itervals_arr = np.array(itervals_list)
        return itervals_arr

    def infer(self, task: ProblemT) -> List[InferenceResult]:
        # itervals_aar: R^{n_task_inner, n_elem_in_lib}
        itervals_arr = self._infer_iteration_num(task)

        # nits_min: R^{n_desc_inner}
        nits_min = np.min(itervals_arr, axis=0)

        # indices_min: R^{n_desc_inner}
        indices_min = np.argmin(itervals_arr, axis=0)

        result_list = []
        for nit, idx in zip(nits_min, indices_min):
            init_solution = self.init_solutions[idx]
            assert init_solution is not None
            res = self.InferenceResult(nit, idx, init_solution)
            result_list.append(res)
        return result_list

    def success_iter_threshold(self) -> float:
        return self.solver_config.n_max_call

    def measure_full_coverage(
        self, tasks: List[ProblemT], solver: BatchProblemSolver
    ) -> CoverageResult:
        logger.info("**compute est values")
        iterval_est_list = []
        init_solutions_est_list = []
        for task in tqdm.tqdm(tasks):
            infer_results = self.infer(task)
            iterval_est_list.extend([res.nit for res in infer_results])  # NOTE: flatten!
            init_solutions_est_list.append([res.init_solution for res in infer_results])
        logger.info("**compute real values")

        resultss = solver.solve_batch(tasks, init_solutions_est_list)  # type: ignore

        iterval_real_list = []
        for results in resultss:
            for result in results:
                iterval_real_list.append(get_clamped_iter(result, self.solver_config))

        success_iter = self.success_iter_threshold()
        coverage_result = CoverageResult(
            np.array(iterval_real_list), np.array(iterval_est_list), success_iter
        )
        logger.info(coverage_result)
        return coverage_result

    def measure_coverage(self, tasks: List[ProblemT]) -> float:
        threshold = self.success_iter_threshold()
        total_count = 0
        success_count = 0
        for task in tasks:
            infer_res_list = self.infer(task)
            for infer_res in infer_res_list:
                total_count += 1
                if infer_res.nit < threshold:
                    success_count += 1
        return success_count / total_count

    def dump(self, base_path: Path) -> None:
        cpu_device = torch.device("cpu")
        copied = copy.deepcopy(self)

        if copied.ae_model_shared is not None:
            copied.ae_model_shared.put_on_device(cpu_device)
        for pred in copied.predictors:
            pred.put_on_device(cpu_device)

        name = "Library-{}-{}.pkl".format(self.task_type.__name__, self.uuidval)
        file_path = base_path / name
        with file_path.open(mode="wb") as f:
            pickle.dump(copied, f)
        logger.info("dumped library to {}".format(file_path))

        name = "MetaData-{}-{}.json".format(self.task_type.__name__, self.uuidval)
        file_path = base_path / name
        if not file_path.exists():
            with file_path.open(mode="w") as f:
                json.dump(self.meta_data, f, indent=4)

    @classmethod
    def load(
        cls,
        base_path: Path,
        task_type: Type[ProblemT],
        solver_type: Type[AbstractScratchSolver[ConfigT, ResultT]],
        device: Optional[torch.device] = None,
        check_hash: bool = True,
    ) -> List["SolutionLibrary[ProblemT, ConfigT, ResultT]"]:
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        libraries = []
        library_paths = []
        for path in base_path.iterdir():
            # logger.debug("load from path: {}".format(path))
            m = re.match(r"Library-(\w+)-(\w+).pkl", path.name)
            if m is not None and m[1] == task_type.__name__:
                logger.debug("library found at {}".format(path))
                library_paths.append(path)

        latest_path = None
        latest_timestamp = None
        for path in library_paths:
            timestamp = path.stat().st_ctime

            readable_timestamp = datetime.datetime.fromtimestamp(timestamp)
            logger.debug("lib: {}, ts: {}".format(path, readable_timestamp))
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_path = path
        assert latest_path is not None

        task_hash = task_type.compute_distribution_hash()

        logger.debug("load latest library from {}".format(latest_path))
        with latest_path.open(mode="rb") as f:
            lib: "SolutionLibrary[ProblemT, ConfigT, ResultT]" = pickle.load(f)
            assert lib.device == torch.device("cpu")
            for pred in lib.predictors:
                assert not pred.training
            lib.put_on_device(device)
            # In most case, user will use the library in a single process
            # thus we dont need to care about thread stuff.
            lib.limit_thread = False  # faster
            if check_hash:
                # check if lib.task_distribution_hash attribute exists
                # if not, it means that the library is old version
                # and we cannot check the hash
                if hasattr(lib, "task_distribution_hash"):
                    if lib.task_distribution_hash != task_hash:
                        msg = f"task_distribution_hash mismatch: {lib.task_distribution_hash} != {task_hash}\n"
                        msg += "task definition has been change after training the library."
                        raise RuntimeError(msg)
                else:
                    logger.warning("cannot check hash because library is old version")
            libraries.append(lib)
        return libraries  # type: ignore

    def unbundle(self) -> List["SolutionLibrary[ProblemT, ConfigT, ResultT]"]:
        # split into list of singleton libraries
        singleton_list = []
        for predictor, init_solution, margin in zip(
            self.predictors, self.init_solutions, self.margins
        ):
            singleton = SolutionLibrary(
                task_type=self.task_type,
                task_distribution_hash=self.task_distribution_hash,
                solver_type=self.solver_type,
                solver_config=self.solver_config,
                ae_model_shared=self.ae_model_shared,
                predictors=[predictor],
                init_solutions=[init_solution],
                margins=[margin],
                uuidval="dummy",
                meta_data={},
            )
            singleton_list.append(singleton)
        return singleton_list

    @classmethod
    def from_singletons(
        cls, singletons: List["SolutionLibrary[ProblemT, ConfigT, ResultT]"]
    ) -> "SolutionLibrary[ProblemT, ConfigT, ResultT]":
        singleton = singletons[0]
        predictors = [e.predictors[0] for e in singletons]
        init_solutions = [e.init_solutions[0] for e in singletons]
        margins = [e.margins[0] for e in singletons]

        library = SolutionLibrary(
            task_type=singleton.task_type,
            task_distribution_hash=singleton.task_distribution_hash,
            solver_type=singleton.solver_type,
            solver_config=singleton.solver_config,
            ae_model_shared=singleton.ae_model_shared,
            predictors=predictors,
            init_solutions=init_solutions,
            margins=margins,
            uuidval="dummy",
            meta_data={},
        )
        return library

    def get_singleton(self, idx: int) -> "SolutionLibrary[ProblemT, ConfigT, ResultT]":
        singleton = SolutionLibrary(
            task_type=self.task_type,
            task_distribution_hash=self.task_distribution_hash,
            solver_type=self.solver_type,
            solver_config=self.solver_config,
            ae_model_shared=self.ae_model_shared,
            predictors=[self.predictors[idx]],
            init_solutions=[self.init_solutions[idx]],
            margins=[self.margins[idx]],
            uuidval="dummy",
            meta_data={},
        )
        return singleton


@dataclass
class LibraryBasedSolverBase(AbstractTaskSolver[ProblemT, ConfigT, ResultT]):
    library: SolutionLibrary[ProblemT, ConfigT, ResultT]
    solver: AbstractScratchSolver[ConfigT, ResultT]
    task: Optional[ProblemT]
    timeout: Optional[float]
    previous_false_positive: Optional[bool]
    previous_est_positive: Optional[bool]
    _loginfo_fun: Callable
    _logwarn_fun: Callable

    @classmethod
    def init(
        cls,
        library: SolutionLibrary[ProblemT, ConfigT, ResultT],
        config: Optional[ConfigT] = None,
        use_rospy_logger: bool = False,
    ) -> "LibraryBasedSolverBase[ProblemT, ConfigT, ResultT]":
        if config is None:
            config = library.solver_config
        # internal solver's timeout must be None
        # because inference time must be considered in timeout for fairness
        timeout_stashed = config.timeout  # stash this
        config.timeout = None
        solver = library.solver_type.init(config)

        if use_rospy_logger:
            import rospy

            # NOTE: don't know why but importing rospy at the top of the file
            # cause issue in logging in the training phase. That's why I import
            # it here.
            loginfo_fun = rospy.loginfo
            logwarn_fun = rospy.logwarn
        else:
            loginfo_fun = logger.info
            logwarn_fun = logger.warning

        return cls(library, solver, None, timeout_stashed, None, None, loginfo_fun, logwarn_fun)

    def setup(self, task: ProblemT) -> None:
        assert task.n_inner_task == 1
        problems = [p for p in task.export_problems()]
        self.solver.setup(problems[0])
        self.task = task

    def solve(self) -> ResultT:
        # NOTE: almost copied from skmp.solver.interface
        ts = time.time()

        if self.timeout is not None:
            assert self.timeout > 0

            def handler(sig, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, self.timeout)
            set_ompl_random_seed(0)  # to make result reproducible
        try:
            ret = self._solve()
        except TimeoutError:
            ret = self.solver.get_result_type().abnormal()

        if self.timeout is not None:
            signal.alarm(0)  # reset alarm

        ret.time_elapsed = time.time() - ts
        return ret

    @abstractmethod
    def _solve(self) -> ResultT:
        ...


@dataclass
class LibraryBasedGuaranteedSolver(LibraryBasedSolverBase[ProblemT, ConfigT, ResultT]):
    def _solve(self) -> ResultT:
        self.previous_est_positive = None
        self.previous_false_positive = None

        ts = time.time()
        assert self.task is not None
        inference_results = self.library.infer(self.task)
        assert len(inference_results) == 1
        inference_result = inference_results[0]

        seems_infeasible = inference_result.nit > self.library.success_iter_threshold()
        self._loginfo_fun(
            f"nit {inference_result.nit}: the {self.library.success_iter_threshold()}"
        )
        if seems_infeasible:
            self._logwarn_fun("seems infeasible")
            result_type = self.solver.get_result_type()
            res = result_type.abnormal()
            res.time_elapsed = None
            self.previous_est_positive = False
            self.previous_false_positive = False
            return res
        solver_result = self.solver.solve(inference_result.init_solution)
        solver_result.time_elapsed = time.time() - ts

        self.previous_est_positive = True
        self.previous_false_positive = solver_result.traj is None
        return solver_result


@dataclass
class LibraryBasedHeuristicSolver(LibraryBasedSolverBase[ProblemT, ConfigT, ResultT]):
    def _solve(self) -> ResultT:
        ts = time.time()
        assert self.task is not None
        inference_results = self.library.infer(self.task)
        assert len(inference_results) == 1
        inference_result = inference_results[0]
        solver_result = self.solver.solve(inference_result.init_solution)
        solver_result.time_elapsed = time.time() - ts
        return solver_result


@dataclass
class DifficultProblemPredicate(Generic[ProblemT, ConfigT, ResultT]):
    task_type: Type[ProblemT]
    library: SolutionLibrary[ProblemT, ConfigT, ResultT]
    th_min_iter: float
    th_max_iter: Optional[float] = None

    def __post_init__(self):
        # note: library must be put on cpu
        # to copy into forked processes
        self.library = copy.deepcopy(self.library)
        self.library.put_on_device(torch.device("cpu"))

    def __call__(self, task: ProblemT) -> bool:
        assert task.n_inner_task == 1
        infer_res = self.library.infer(task)[0]
        iterval = infer_res.nit
        if iterval < self.th_min_iter:
            return False
        if self.th_max_iter is None:
            return True
        else:
            return iterval < self.th_max_iter


@dataclass
class LibrarySamplerConfig:
    # you have to tune
    sampling_number_factor: float = 5000
    acceptable_false_positive_rate: float = 0.1

    # maybe you have to tune maybe ...
    inc_coef_mult_snf: float = 1.1  # snf stands for sampling_number_factor
    threshold_inc_snf: float = 0.2  # if gain < expected * this, then increase snf
    n_solution_candidate: int = 100
    n_difficult: int = 500
    n_problem_max: int = 1000000

    # same for all settings (you dont have to tune)
    n_problem_inner: int = 1  # this should be 1 always (2024/02/24)
    sample_from_difficult_region: bool = True
    train_config: TrainConfig = TrainConfig()
    ignore_useless_traj: bool = True
    iterpred_model_config: Optional[Dict] = None
    n_validation: int = 10000
    n_validation_inner: int = 1
    n_determine_batch: int = 2000
    candidate_sample_scale: int = 4
    train_with_encoder: bool = False
    tmp_n_max_call_mult_factor: float = 1.5


@dataclass
class SimpleSolutionLibrarySampler(Generic[ProblemT, ConfigT, ResultT]):
    problem_type: Type[ProblemT]
    library: SolutionLibrary[ProblemT, ConfigT, ResultT]
    config: LibrarySamplerConfig
    pool_single: ProblemPool[ProblemT]
    pool_multiple: ProblemPool[ProblemT]
    problems_validation: List[ProblemT]
    solver: BatchProblemSolver
    sampler: BatchProblemSampler
    determinant: BatchMarginsDeterminant
    test_false_positive_rate: bool
    delete_cache: bool
    project_path: Path
    sampler_state: ActiveSamplerState
    ae_model_pretrained: Optional[
        AutoEncoderBase
    ] = None  # train iteration predctor combined with encoder. Thus ae will no be shared.
    presampled_train_problems: Optional[List[ProblemT]] = None

    @property
    def solver_type(self) -> Type[AbstractScratchSolver[ConfigT, ResultT]]:
        return self.library.solver_type

    @property
    def solver_config(self) -> ConfigT:
        return self.library.solver_config

    def __post_init__(self):
        self.reset_pool()

    @property
    def train_pred_with_encoder(self) -> bool:
        return self.ae_model_pretrained is not None

    @cached_property
    def debug_data_parent_path(self) -> Path:
        path = Path("/tmp") / "hifuku-debug-data"
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()
        return path

    def reset_pool(self) -> None:
        logger.info("resetting pool")
        self.pool_single.reset()
        self.pool_multiple.reset()

    def at_first_iteration(self) -> bool:
        return len(self.library.predictors) == 0

    @classmethod
    def initialize(
        cls,
        problem_type: Type[ProblemT],
        solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]],
        solver_config: ConfigT,
        ae_model: AutoEncoderBase,
        config: LibrarySamplerConfig,
        project_path: Path,
        pool_single: Optional[ProblemPool[ProblemT]] = None,
        pool_multiple: Optional[ProblemPool[ProblemT]] = None,
        problems_validation: Optional[List[ProblemT]] = None,
        solver: Optional[BatchProblemSolver[ConfigT, ResultT]] = None,
        sampler: Optional[BatchProblemSampler[ProblemT]] = None,
        use_distributed: bool = False,
        reuse_cached_validation_set: bool = False,
        test_false_positive_rate: bool = False,
        delete_cache: bool = False,
        n_limit_batch_solver: Optional[int] = None,
        presample_train_problems: bool = False,
    ) -> "SimpleSolutionLibrarySampler[ProblemT, ConfigT, ResultT]":
        """
        use will be used only if either of solver and sampler is not set
        """

        frame = inspect.currentframe()
        assert frame is not None
        _, _, _, values = inspect.getargvalues(frame)
        logger.info("arg of initialize: {}".format(values))

        meta_data = asdict(config)
        library = SolutionLibrary.initialize(
            problem_type,
            solver_t,
            solver_config,
            None if config.train_with_encoder else ae_model,
            meta_data,
        )
        sampler_state = ActiveSamplerState(config.sampling_number_factor)

        # setup solver, sampler, determinant
        if solver is None:
            solver = (
                DistributedBatchProblemSolver(
                    solver_t, solver_config, n_limit_batch=n_limit_batch_solver
                )
                if use_distributed
                else MultiProcessBatchProblemSolver(
                    solver_t, solver_config, n_limit_batch=n_limit_batch_solver
                )
            )
        assert solver.solver_t == solver_t
        assert solver.config == solver_config
        if sampler is None:
            sampler = (
                DistributeBatchProblemSampler()
                if use_distributed
                else MultiProcessBatchProblemSampler()
            )
        determinant = (
            DistributeBatchMarginsDeterminant()
            if use_distributed
            else MultiProcesBatchMarginsDeterminant()
        )

        # setup pools
        if pool_single is None:
            logger.info("problem pool is not specified. use SimpleProblemPool")
            pool_single = TrivialProblemPool(problem_type, 1)
        assert pool_single.n_problem_inner == 1

        if pool_multiple is None:
            logger.info("problem pool is not specified. use SimpleProblemPool")
            # TODO: smelling! n_problem_inner should not be set here
            pool_multiple = TrivialProblemPool(problem_type, config.n_problem_inner)
        assert pool_multiple.parallelizable()

        # create validation problems
        project_path.mkdir(exist_ok=True)
        validation_cache_path = project_path / "{}-validation_set.cache".format(
            problem_type.__name__
        )

        if reuse_cached_validation_set:
            assert problems_validation is None
            assert validation_cache_path.exists()
            with validation_cache_path.open(mode="rb") as f:
                problems_validation = pickle.load(f)
                assert problems_validation is not None
            logger.info("validation set is load from {}".format(validation_cache_path))
        else:
            if problems_validation is None:
                logger.info("start creating validation set")

                problems_validation = sampler.sample_batch(
                    config.n_validation,
                    TrivialProblemPool(problem_type, config.n_validation_inner).as_predicated(),
                    delete_cache=delete_cache,
                )

                with validation_cache_path.open(mode="wb") as f:
                    pickle.dump(problems_validation, f)
                logger.info(
                    "validation set with {} elements is created".format(len(problems_validation))
                )
        assert len(problems_validation) > 0

        if presample_train_problems:
            predicated_pool = pool_multiple.as_predicated()
            n_require = config.n_problem_max
            logger.info("presample {} tasks".format(n_require))
            presampled_train_problems = sampler.sample_batch(
                n_require, predicated_pool, delete_cache
            )
        else:
            presampled_train_problems = None

        logger.info("library sampler config: {}".format(config))
        return cls(
            problem_type,
            library,
            config,
            pool_single,
            pool_multiple,
            problems_validation,
            solver,
            sampler,
            determinant,
            test_false_positive_rate,
            delete_cache,
            project_path,
            sampler_state,
            ae_model if config.train_with_encoder else None,
            presampled_train_problems,
        )

    def step_active_sampling(self) -> bool:
        """
        return False if failed
        """
        prof_info = ProfileInfo()
        self.sampler_state.check_consistency()
        assert len(self.sampler_state.coverage_results) == len(self.library.predictors)

        logger.info("active sampling step")
        ts = time.time()

        ts_determine_cand = time.time()
        init_solution, gain_expected = self._determine_init_solution(self.config.n_difficult)
        logger.info(f"sampling nuber factor: {self.sampler_state.sampling_number_factor}")
        n_problem_now = int((1.0 / gain_expected) * self.sampler_state.sampling_number_factor)
        logger.info(f"n_problem_now: {n_problem_now}")
        prof_info.t_determine_cand = time.time() - ts_determine_cand

        with TemporaryDirectory() as td:
            self.reset_pool()
            predictor = self._train_predictor(
                init_solution, self.project_path, n_problem_now, prof_info
            )

            ts_margin = time.time()
            ret = self._determine_margins(predictor, init_solution)
            prof_info.t_margin = time.time() - ts_margin

        if ret is None:
            logger.info("determine margin failed. returning None")
            elapsed_time = time.time() - ts
            logger.info("elapsed time in active sampling: {} min".format(elapsed_time / 60.0))
            logger.info("prof_info: {}".format(prof_info))
            prof_info.t_total = elapsed_time
            self.sampler_state.elapsed_time_history.append(prof_info)
            coverage_est = self.sampler_state.coverage_est_history[-1]
            self.sampler_state.failure_count += 1
        else:
            margins, coverage_result, coverage_est = ret
            logger.info("margin for latest iterpred is {}".format(margins[-1]))
            logger.debug("determined margins {}".format(margins))

            # update library
            self.library.predictors.append(predictor)
            self.library.init_solutions.append(init_solution)
            self.library.margins = margins

            coverage_est = self.library.measure_coverage(self.problems_validation)  # double check
            assert np.abs(coverage_est - coverage_est) < 1e-6  # doulbe check
            prof_info.t_total = time.time() - ts

            self.sampler_state.coverage_results.append(coverage_result)
            self.sampler_state.margins_history.append(copy.deepcopy(margins))

        self.sampler_state.elapsed_time_history.append(prof_info)
        self.sampler_state.coverage_est_history.append(coverage_est)

        if len(self.sampler_state.coverage_est_history) > 0:
            coverage_previous = self.sampler_state.coverage_est_history[-1]
            if (coverage_est - coverage_previous) < gain_expected * self.config.threshold_inc_snf:
                self.sampler_state.sampling_number_factor *= self.config.inc_coef_mult_snf
                logger.info(
                    f"expected gain is {gain_expected}, but actual gain is {coverage_est - coverage_previous}. increase sampling number factor to {self.sampler_state.sampling_number_factor}"
                )

        logger.info("elapsed time in active sampling: {} min".format(prof_info.t_total / 60.0))
        logger.info("prof_info: {}".format(prof_info))
        t_total_list = [e.t_total for e in self.sampler_state.elapsed_time_history]
        logger.info("current elapsed time history: {}".format(t_total_list))
        logger.info(
            "current coverage est history: {}".format(self.sampler_state.coverage_est_history)
        )
        self.library.dump(self.project_path)
        self.sampler_state.dump(self.project_path)
        return True

    @property
    def difficult_iter_threshold(self) -> float:
        return self.library.solver_config.n_max_call

    def _determine_margins(
        self,
        predictor: Union[IterationPredictorWithEncoder, IterationPredictor],
        init_solution: Trajectory,
    ) -> Optional[Tuple[List[float], CoverageResult, float]]:
        # TODO: move this whole "adjusting" operation to a different method
        logger.info("start measuring coverage")
        singleton_library = SolutionLibrary(
            task_type=self.problem_type,
            task_distribution_hash=self.library.task_distribution_hash,
            solver_type=self.solver_type,
            solver_config=self.solver_config,
            ae_model_shared=self.library.ae_model_shared,
            predictors=[predictor],
            init_solutions=[init_solution],
            margins=[0.0],
            uuidval="dummy",
            meta_data={},
        )
        coverage_result = singleton_library.measure_full_coverage(
            self.problems_validation, self.solver
        )
        logger.info(coverage_result)

        if len(self.library.predictors) > 0:
            # determine margin using cmaes
            cma_std = self.solver_config.n_max_call * 0.5
            coverages_new = self.sampler_state.coverage_results + [coverage_result]
            coverage_est_last = self.sampler_state.coverage_est_history[-1]
            results = self.determinant.determine_batch(
                self.config.n_determine_batch,
                coverages_new,
                self.solver_config.n_max_call,
                self.config.acceptable_false_positive_rate,
                cma_std,
                minimum_coverage=coverage_est_last,
            )

            best_margins = None
            max_coverage = coverage_est_last
            for result in results:
                if result is None:
                    continue
                if result.coverage > max_coverage:
                    max_coverage = result.coverage
                    best_margins = result.best_margins

            if best_margins is None:
                return None
            margins = best_margins
        else:
            logger.info("determine margin using exact method")
            margin, max_coverage = coverage_result.determine_margin(
                self.config.acceptable_false_positive_rate
            )

            if not np.isfinite(margin) and self.config.ignore_useless_traj:
                message = "margin value {} is invalid. retrun from active_sampling".format(margin)
                logger.info(message)
                return None
            margins = [margin]

        assert max_coverage is not None
        logger.info("optimal coverage estimate is set to {}".format(max_coverage))
        return margins, coverage_result, max_coverage

    def _train_predictor(
        self,
        init_solution: Trajectory,
        project_path: Path,
        n_problem: int,
        profile_info: ProfileInfo,
    ) -> Union[IterationPredictorWithEncoder, IterationPredictor]:
        pp = project_path

        ts_dataset = time.time()
        predicated_pool = self.pool_multiple.as_predicated()

        # NOTE: to my future self: you can't use presampled problems if you'd like to
        # sample tasks using some predicate!!
        logger.info("generate {} tasks".format(n_problem))
        if self.presampled_train_problems is None:
            problems = self.sampler.sample_batch(n_problem, predicated_pool, self.delete_cache)
        else:
            logger.debug("use presampled tasks")
            problems = self.presampled_train_problems[:n_problem]

        logger.info("start generating dataset")
        # create dataset. Dataset creation by a large problem set often causes
        # memory error. To avoid this, we split the batch and process separately
        # if len(problems) > n_tau.
        init_solutions = [init_solution] * len(problems)

        resultss = self.solver.solve_batch(
            problems,
            init_solutions,
            tmp_n_max_call_mult_factor=self.config.tmp_n_max_call_mult_factor,
        )

        use_weighting = False
        if use_weighting:
            # this modification of loss function using cost may be related to the following articles
            # Good introduction:
            # https://machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/
            # A concise review:
            # Haixiang, Guo, et al. "Learning from class-imbalanced data: Review of methods and applications." Expert systems with applications 73 (2017): 220-239.

            # NOTE about the performance
            # the result here suggests that performance is not improved by this modification
            # https://github.com/HiroIshida/hifuku/pull/27
            # https://github.com/HiroIshida/hifuku/issues/28
            # assert False  # 2024/01/22

            # 2024/02/21: changed weighting scheme refering Tanimoto, Akira, et al. "Improving imbalanced classification using near-miss instances." Expert Systems with Applications 201 (2022): 117130.
            # NOTE about the performance: with_weightning
            # [INFO] 2024-02-21 00:52:47,981 hifuku.library.core: current coverage est history: [0.1137, 0.2119, 0.2149, 0.2209, 0.2574, 0.3029, 0.3264, 0.3277, 0.3328, 0.3385, 0.3385, 0.3455]
            # [INFO] 2024-02-21 01:12:58,221 hifuku.library.core: optimal coverage estimate is set to 0.3455
            # [INFO] 2024-02-21 01:35:03,758 hifuku.library.core: optimal coverage estimate is set to 0.3455
            # [INFO] 2024-02-21 01:57:14,569 hifuku.library.core: optimal coverage estimate is set to 0.3455
            # without weighting ...
            # [INFO] 2024-02-20 12:43:03,258 hifuku.library.core: current coverage est history: [0.105, 0.1683, 0.2526, 0.2599, 0.2599, 0.2677, 0.2961, 0.3197, 0.3199, 0.3213, 0.3297, 0.342, 0.3583, 0.3691, 0.4017, 0.466, 0.466, 0.466, 0.4825, 0.4847, 0.4847, 0.5061, 0.5126, 0.5482, 0.5649, 0.6106, 0.6142]
            # Seems that performance rather worse
            # assert False  # 2024/2/21

            # 2024/2/22:
            # Even after modification n_inner = 1, the above tendency is the case (weighting is bit worse than the original).
            # Even worse, as weighting requires infer for each samples, the weight determination becomes particulary costly
            # when n_inner = 1
            assert False, "don't use. Really. (2024/2/23)"
            weights = torch.ones((len(problems), problems[0].n_inner_task))
            n_total = len(problems) * problems[0].n_inner_task

            # actually ...
            def res_to_nit(res: ResultProtocol) -> float:
                if res.traj is not None:
                    return float(res.n_call)
                else:
                    return np.inf

            this_nitss = torch.tensor([[res_to_nit(r) for r in results] for results in resultss])
            solved_by_this = this_nitss < self.library.success_iter_threshold()
            logger.info(f"rate of solved by this: {torch.sum(solved_by_this) / n_total}")

            # compute if each task is difficult or not
            if len(self.library.predictors) > 0:
                infer_resultss = [self.library.infer(task) for task in problems]
                infer_nitss = torch.tensor(
                    [[e.nit for e in infer_results] for infer_results in infer_resultss]
                )
                unsolvable_yet = infer_nitss > self.library.success_iter_threshold()
                logger.info(f"rate of unsolvable yet: {torch.sum(unsolvable_yet) / n_total}")
            else:
                unsolvable_yet = torch.ones(len(problems), problems[0].n_inner_task, dtype=bool)

            # if unsolvable so far but solved by this, such sample is quite valuable for training
            bools_unsolvable_yet_and_solved_by_this = unsolvable_yet & solved_by_this
            n_this = torch.sum(bools_unsolvable_yet_and_solved_by_this)
            logger.info(f"rate of solved by only this: {n_this / n_total}")
            n_other = n_total - n_this

            c_plus = n_total / (n_this * 2)
            c_minus = n_total / (n_other * 2)
            logger.info(f"c_plus: {c_plus}, c_minus: {c_minus}")
            weights[bools_unsolvable_yet_and_solved_by_this] = c_plus
            weights[~bools_unsolvable_yet_and_solved_by_this] = c_minus
        else:
            weights = None

        logger.info("creating dataset")
        dataset = IterationPredictorDataset.construct_from_tasks_and_resultss(
            init_solution,
            problems,
            resultss,
            self.solver_config,
            weights,
            self.library.ae_model_shared,
        )

        profile_info.t_dataset = time.time() - ts_dataset

        logger.info("start training model")
        ts_train = time.time()
        # determine 1dim tensor dimension by temp creation of a problem
        # TODO: should I implement this as a method?
        problem = self.problem_type.sample(1, standard=True)
        table = problem.export_table()
        vector_desc = table.get_vector_descs()[0]
        n_dim_vector_description = vector_desc.shape[0]

        # train
        if self.train_pred_with_encoder:
            assert self.ae_model_pretrained is not None
            n_bottleneck = self.ae_model_pretrained.n_bottleneck
        else:
            assert self.library.ae_model_shared is not None
            n_bottleneck = self.library.ae_model_shared.n_bottleneck

        if self.config.iterpred_model_config is not None:
            iterpred_model_conf = IterationPredictorConfig(
                n_dim_vector_description, n_bottleneck, **self.config.iterpred_model_config
            )
        else:
            iterpred_model_conf = IterationPredictorConfig(n_dim_vector_description, n_bottleneck)

        if self.train_pred_with_encoder:
            assert self.ae_model_pretrained is not None
            iterpred_model = IterationPredictor(iterpred_model_conf)
            ae_model_pretrained = copy.deepcopy(self.ae_model_pretrained)
            ae_model_pretrained.put_on_device(iterpred_model.device)
            assert not isinstance(ae_model_pretrained, NullAutoEncoder)
            # the right above assertion ensure that ae_model_pretrained has a device...
            assert iterpred_model.device == ae_model_pretrained.device  # type: ignore[attr-defined]
            conf = IterationPredictorWithEncoderConfig(iterpred_model, ae_model_pretrained)
            model: Union[
                IterationPredictorWithEncoder, IterationPredictor
            ] = IterationPredictorWithEncoder(conf)
        else:
            model = IterationPredictor(iterpred_model_conf)

        tcache = TrainCache.from_model(model)

        def is_stoppable(tcache: TrainCache) -> bool:
            valid_losses = tcache.reduce_to_lossseq(tcache.validate_lossseq_table)
            n_step = len(valid_losses)
            idx_min = np.argmin(valid_losses)
            t_acceptable = 20
            no_improvement_for_long = bool((n_step - idx_min) > t_acceptable)
            return no_improvement_for_long

        train(pp, tcache, dataset, self.config.train_config, is_stoppable=is_stoppable)
        model.eval()
        profile_info.t_train = time.time() - ts_train
        return model

    def _sample_solution_canidates(
        self,
        n_sample: int,
        problem_pool: ProblemPool[ProblemT],
    ) -> List[Trajectory]:

        assert problem_pool.n_problem_inner == 1

        if self.config.sample_from_difficult_region and not self.at_first_iteration():
            # because sampling from near-feasible-boundary is effective in most case....
            pred_bit_difficult = DifficultProblemPredicate(
                problem_pool.problem_type,
                self.library,
                self.difficult_iter_threshold,
                self.difficult_iter_threshold * 1.2,
            )
            predicated_pool_bit_difficult = problem_pool.make_predicated(pred_bit_difficult, 40)

            # but, we also need to sample from far-boundary because some of the possible
            # feasible regions are disjoint from the ones obtained so far
            pred_difficult = DifficultProblemPredicate(
                problem_pool.problem_type, self.library, self.difficult_iter_threshold, None
            )
            predicated_pool_difficult = problem_pool.make_predicated(pred_difficult, 40)
        else:
            # "sampling from difficult" get stuck when the classifier is not properly trained
            # which becomes problem in test where classifier is trained with dummy small sample
            predicated_pool_bit_difficult = problem_pool.as_predicated()
            predicated_pool_difficult = predicated_pool_bit_difficult

        prefix = "_sample_solution_canidates:"

        logger.info("{} start sampling solution solved difficult problems".format(prefix))

        # TODO: dont hardcode
        n_batch = n_sample * self.config.candidate_sample_scale
        n_batch_little_difficult = int(n_batch * 0.5)
        n_batch_difficult = n_batch - n_batch_little_difficult

        feasible_solutions: List[Trajectory] = []
        while True:
            logger.info("{} sample batch".format(prefix))
            problems1 = self.sampler.sample_batch(
                n_batch_little_difficult, predicated_pool_bit_difficult
            )
            problems2 = self.sampler.sample_batch(n_batch_difficult, predicated_pool_difficult)
            problems = problems1 + problems2
            for prob in problems:
                prob.delete_cache()

            # NOTE: shuffling is required asin the following sectino, for loop is existed
            # as soon as number of candidates exceeds n_sample
            # we need to "mixutre" bit-difficult and difficult problems
            random.shuffle(problems)

            logger.info("{} solve batch".format(prefix))
            resultss = self.solver.solve_batch(problems, [None] * n_batch, use_default_solver=True)

            for results in resultss:
                result = results[0]
                if result.traj is not None:
                    feasible_solutions.append(result.traj)
                    if len(feasible_solutions) == n_sample:
                        return feasible_solutions
            logger.info("{} progress {} / {} ".format(prefix, len(feasible_solutions), n_sample))

    def _sample_difficult_problems(
        self,
        n_sample: int,
        problem_pool: ProblemPool[ProblemT],
    ) -> Tuple[List[ProblemT], List[ProblemT]]:

        difficult_problems: List[ProblemT] = []
        easy_problems: List[ProblemT] = []
        with tqdm.tqdm(total=n_sample) as pbar:
            while len(difficult_problems) < n_sample:
                logger.debug("try sampling difficutl problem...")
                problem = next(problem_pool)
                assert problem.n_inner_task == 1
                infer_res = self.library.infer(problem)[0]
                iterval = infer_res.nit
                is_difficult = iterval > self.difficult_iter_threshold
                if is_difficult:
                    logger.debug("sampled! number: {}".format(len(difficult_problems)))
                    difficult_problems.append(problem)
                    pbar.update(1)
                else:
                    easy_problems.append(problem)
        return difficult_problems, easy_problems

    def _select_solution_candidates(
        self, candidates: List[Trajectory], problems: List[ProblemT]
    ) -> Tuple[Trajectory, int]:
        logger.info("select single solution out of {} candidates".format(len(candidates)))

        candidates_repeated = []
        n_problem = len(problems)
        n_cand = len(candidates)
        for cand in candidates:
            candidates_repeated.extend([cand] * n_problem)
        problems_repeated = problems * n_cand
        resultss = self.solver.solve_batch(problems_repeated, candidates_repeated)
        assert len(resultss) == n_problem * n_cand

        def split_list(lst, n):
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        resultss_list = split_list(
            resultss, n_problem
        )  # split the result such that each list corresponds to candidate trajectory

        n_solved_max = 0
        best_cand: Optional[Trajectory] = None
        for idx_cand, resultss in enumerate(resultss_list):
            n_solved = sum([results[0].traj is not None for results in resultss])
            logger.info("cand_idx {}: {}".format(idx_cand, n_solved))
            if n_solved > n_solved_max:
                n_solved_max = n_solved
                best_cand = candidates[idx_cand]
        logger.info("n_solved_max of candidates: {}".format(n_solved_max))
        assert best_cand is not None
        return best_cand, n_solved_max

    def _determine_init_solution(self, n_difficult: int) -> Tuple[Trajectory, float]:
        n_repeat_budget = 2
        for i_repeat in range(n_repeat_budget):
            logger.info("sample solution candidates ({}-th repeat)".format(i_repeat))
            problem_pool = self.pool_single
            solution_candidates = self._sample_solution_canidates(
                self.config.n_solution_candidate, problem_pool
            )
            self.sampler_state.candidates_history.append(solution_candidates)

            logger.info("sample difficult problems")
            if self.at_first_iteration():
                difficult_problems = [next(problem_pool) for _ in range(n_difficult)]
                n_total = len(difficult_problems)
            else:
                difficult_problems, easy_problems = self._sample_difficult_problems(
                    n_difficult, problem_pool
                )
                n_total = len(difficult_problems) + len(easy_problems)

            best_solution, n_solved_max = self._select_solution_candidates(
                solution_candidates, difficult_problems
            )
            # the rate of solved difficult problems / all
            rate_difficult_solved = n_solved_max / n_total
            if best_solution is not None:
                logger.info(f"rate of difficult problems solved: {rate_difficult_solved}")
                logger.info("found best solution")
                return best_solution, rate_difficult_solved
        raise RuntimeError("consumed all repeat budget")
