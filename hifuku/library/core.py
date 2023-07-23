import copy
import datetime
import json
import logging
import pickle
import random
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Generic, List, Optional, Tuple, Type, Union

import numpy as np
import threadpoolctl
import torch
import tqdm
from mohou.trainer import TrainCache, TrainConfig, train
from rpbench.interface import AbstractTaskSolver
from skmp.solver.interface import AbstractScratchSolver, ConfigT, ResultT
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
class SolutionLibrary(Generic[ProblemT, ConfigT, ResultT]):
    """Solution Library

    limitting threadnumber takes nonnegligible time. So please set
    limit_thread = false in performance evaluation time. However,
    when in attempt to run in muliple process, one must set it True.
    """

    task_type: Type[ProblemT]
    solver_type: Type[AbstractScratchSolver[ConfigT, ResultT]]
    solver_config: ConfigT
    ae_model_shared: Optional[
        AutoEncoderBase
    ]  # if None, when each predictor does not share autoencoder
    predictors: List[Union[IterationPredictor, IterationPredictorWithEncoder]]
    margins: List[float]
    coverage_results: Optional[List[CoverageResult]]
    solvable_threshold_factor: float
    uuidval: str
    meta_data: Dict
    limit_thread: bool = False
    _margins_history: Optional[List[List[float]]] = None
    _candidates_history: Optional[List[List[Trajectory]]] = None
    _optimal_coverage_estimate: Optional[
        float
    ] = None  # the cached optimal coverage after margins optimization

    def __setstate__(self, state):
        # NOTE: for backward compatibility
        is_old_version = "ae_model" in state
        if is_old_version:
            assert "ae_model_shared" not in state
            state["ae_model_shared"] = state["ae_model"]
            del state["ae_model"]
            assert "ae_model" not in state
        self.__dict__.update(state)

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
        solvable_threshold_factor: float,
        meta_data: Optional[Dict] = None,
    ) -> "SolutionLibrary[ProblemT, ConfigT, ResultT]":
        uuidval = str(uuid.uuid4())[-8:]
        if meta_data is None:
            meta_data = {}
        return cls(
            task_type,
            solver_type,
            config,
            ae_model,
            [],
            [],
            [],
            solvable_threshold_factor,
            uuidval,
            meta_data,
            True,  # assume that we are gonna build library and not in eval time.
            [],
            [],
            None,
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
            init_solution = self.predictors[idx].initial_solution
            assert init_solution is not None
            res = self.InferenceResult(nit, idx, init_solution)
            result_list.append(res)
        return result_list

    def success_iter_threshold(self) -> float:
        threshold = self.solver_config.n_max_call * self.solvable_threshold_factor
        return threshold

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

        logger.debug("load latest library from {}".format(latest_path))
        with latest_path.open(mode="rb") as f:
            lib: "SolutionLibrary[ProblemT, ConfigT, ResultT]" = pickle.load(f)
            assert lib.device == torch.device("cpu")
            lib.put_on_device(device)
            # In most case, user will use the library in a single process
            # thus we dont need to care about thread stuff.
            lib.limit_thread = False  # faster
            libraries.append(lib)
        return libraries  # type: ignore

    def unbundle(self) -> List["SolutionLibrary[ProblemT, ConfigT, ResultT]"]:
        # split into list of singleton libraries
        singleton_list = []
        for predictor, margin in zip(self.predictors, self.margins):
            assert predictor.initial_solution is not None

            singleton = SolutionLibrary(
                task_type=self.task_type,
                solver_type=self.solver_type,
                solver_config=self.solver_config,
                ae_model_shared=self.ae_model_shared,
                predictors=[predictor],
                margins=[margin],
                coverage_results=None,
                solvable_threshold_factor=self.solvable_threshold_factor,
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
        margins = [e.margins[0] for e in singletons]

        library = SolutionLibrary(
            task_type=singleton.task_type,
            solver_type=singleton.solver_type,
            solver_config=singleton.solver_config,
            ae_model_shared=singleton.ae_model_shared,
            predictors=predictors,
            margins=margins,
            coverage_results=None,
            solvable_threshold_factor=singleton.solvable_threshold_factor,
            uuidval="dummy",
            meta_data={},
        )
        return library

    def get_singleton(self, idx: int) -> "SolutionLibrary[ProblemT, ConfigT, ResultT]":
        singleton = SolutionLibrary(
            task_type=self.task_type,
            solver_type=self.solver_type,
            solver_config=self.solver_config,
            ae_model_shared=self.ae_model_shared,
            predictors=[self.predictors[idx]],
            margins=[self.margins[idx]],
            coverage_results=None,
            solvable_threshold_factor=self.solvable_threshold_factor,
            uuidval="dummy",
            meta_data={},
        )
        return singleton


@dataclass
class LibraryBasedSolverBase(AbstractTaskSolver[ProblemT, ConfigT, ResultT]):
    library: SolutionLibrary[ProblemT, ConfigT, ResultT]
    solver: AbstractScratchSolver[ConfigT, ResultT]
    task: Optional[ProblemT]
    previous_false_positive: Optional[bool]

    @classmethod
    def init(
        cls, library: SolutionLibrary[ProblemT, ConfigT, ResultT]
    ) -> "LibraryBasedSolverBase[ProblemT, ConfigT, ResultT]":
        solver = library.solver_type.init(library.solver_config)
        return cls(library, solver, None, None)

    def setup(self, task: ProblemT) -> None:
        assert task.n_inner_task == 1
        p = task.export_problems()[0]
        self.solver.setup(p)
        self.task = task


@dataclass
class LibraryBasedGuaranteedSolver(LibraryBasedSolverBase[ProblemT, ConfigT, ResultT]):
    def solve(self) -> ResultT:
        self.previous_false_positive = None

        ts = time.time()
        assert self.task is not None
        inference_results = self.library.infer(self.task)
        assert len(inference_results) == 1
        inference_result = inference_results[0]

        seems_infeasible = inference_result.nit > self.library.success_iter_threshold()
        if seems_infeasible:
            result_type = self.solver.get_result_type()
            return result_type.abnormal()
        solver_result = self.solver.solve(inference_result.init_solution)
        solver_result.time_elapsed = time.time() - ts

        self.previous_false_positive = solver_result.traj is None
        return solver_result


@dataclass
class LibraryBasedHeuristicSolver(LibraryBasedSolverBase[ProblemT, ConfigT, ResultT]):
    def solve(self) -> ResultT:
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
    n_problem: int
    n_problem_inner: int
    train_config: TrainConfig
    n_solution_candidate: int = 10
    n_difficult_problem: int = 100
    solvable_threshold_factor: float = 0.8
    difficult_threshold_factor: float = 0.8  # should equal to solvable_threshold_factor
    acceptable_false_positive_rate: float = 0.005
    sample_from_difficult_region: bool = (
        True  # In test, classifier cannot be wel trained. So this should be False
    )
    ignore_useless_traj: bool = True
    iterpred_model_config: Optional[Dict] = None
    bootstrap_trial: int = 0
    bootstrap_percentile: float = 95.0
    n_validation: int = 1000
    n_validation_inner: int = 10
    n_determine_batch: int = 80
    candidate_sample_scale: int = 10
    train_with_encoder: bool = False


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
    adjust_margins: bool
    invalidate_gridsdf: bool
    project_path: Path
    ae_model_pretrained: Optional[
        AutoEncoderBase
    ] = None  # train iteration predctor combined with encoder. Thus ae will no be shared.

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
        adjust_margins: bool = True,
        invalidate_gridsdf: bool = False,
        n_limit_batch_solver: Optional[int] = None,
    ) -> "SimpleSolutionLibrarySampler[ProblemT, ConfigT, ResultT]":
        """
        use will be used only if either of solver and sampler is not set
        """

        meta_data = asdict(config)
        library = SolutionLibrary.initialize(
            problem_type,
            solver_t,
            solver_config,
            None if config.train_with_encoder else ae_model,
            config.solvable_threshold_factor,
            meta_data,
        )

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
        validation_cache_path = project_path / "{}-validation_set.pkl".format(problem_type.__name__)

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
                if config.n_problem_inner == 1:
                    message = "In almost all case (except user's special intension), inner = 1 means that world description perfectly determines the problem. Therefore, setting n_validation_inner>1 is waste of computational time"
                    assert config.n_validation_inner == 1, message
                problems_validation = sampler.sample_batch(
                    config.n_validation,
                    TrivialProblemPool(problem_type, config.n_validation_inner).as_predicated(),
                    invalidate_gridsdf=invalidate_gridsdf,
                )

                with validation_cache_path.open(mode="wb") as f:
                    pickle.dump(problems_validation, f)
                logger.info(
                    "validation set with {} elements is created".format(len(problems_validation))
                )
        assert len(problems_validation) > 0

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
            adjust_margins,
            invalidate_gridsdf,
            project_path,
            ae_model if config.train_with_encoder else None,
        )

    def _generate_problem_samples(self) -> List[ProblemT]:
        predicated_pool = self.pool_multiple.as_predicated()
        problems = self.sampler.sample_batch(
            self.config.n_problem, predicated_pool, self.invalidate_gridsdf
        )
        # logger.info("use stratified sampling to generate problem set")
        # # stratified sampling
        # th = self.difficult_iter_threshold
        # interval_list = [(0, th), (th, th * 1.2), (th * 1.2, None)]
        # n_problem_list = split_number(self.config.n_problem, len(interval_list))

        # problems = []
        # for interval, n_problem in zip(interval_list, n_problem_list):
        #     logger.info("interval: {}, n_problem: {}".format(interval, n_problem))
        #     predicate = DifficultProblemPredicate(
        #         self.problem_type, self.library, interval[0], interval[1]
        #     )
        #     predicated_pool = self.pool_multiple.make_predicated(predicate, 10)
        #     problems_part = self.sampler.sample_batch(
        #         n_problem, predicated_pool, self.invalidate_gridsdf
        #     )
        #     problems.extend(problems_part)
        # assert len(problems) == self.config.n_problem
        return problems

    def step_active_sampling(self) -> None:
        logger.info("active sampling step")
        self.reset_pool()

        init_solution = self._determine_init_solution()
        problems = self._generate_problem_samples()
        predictor = self.learn_predictor(init_solution, self.project_path, problems)

        logger.info("start measuring coverage")
        singleton_library = SolutionLibrary(
            task_type=self.problem_type,
            solver_type=self.solver_type,
            solver_config=self.solver_config,
            ae_model_shared=self.library.ae_model_shared,
            predictors=[predictor],
            margins=[0.0],
            coverage_results=None,
            solvable_threshold_factor=self.config.solvable_threshold_factor,
            uuidval="dummy",
            meta_data={},
        )
        coverage_result = singleton_library.measure_full_coverage(
            self.problems_validation, self.solver
        )
        logger.info(coverage_result)

        debug_file_path = self.debug_data_parent_path / "hifuku_coverage_debug.pkl"
        with debug_file_path.open(mode="wb") as f:
            pickle.dump(coverage_result, f)

        assert self.library.coverage_results is not None

        if not self.adjust_margins:
            margins = self.library.margins + [0.0]
        else:
            if len(self.library.predictors) > 0:
                # determine margin using cmaes
                cma_std = self.solver_config.n_max_call * 0.5
                coverages_new = self.library.coverage_results + [coverage_result]
                logger.info(
                    "optimal coverage estimate is set to {}".format(
                        self.library._optimal_coverage_estimate
                    )
                )
                results = self.determinant.determine_batch(
                    self.config.n_determine_batch,
                    coverages_new,
                    self.solver_config.n_max_call,
                    self.config.acceptable_false_positive_rate,
                    cma_std,
                    minimum_coverage=self.library._optimal_coverage_estimate,
                )

                best_margins = None
                max_coverage = -np.inf
                for result in results:
                    if result is None:
                        continue
                    if result.coverage > max_coverage:
                        max_coverage = result.coverage
                        best_margins = result.best_margins

                if best_margins is None:
                    # TODO: we should not ignore when self.config.ignore_useless_traj=False
                    logger.info("no improvement by this element")
                    return

                margins = best_margins
                self.library._optimal_coverage_estimate = max_coverage
                logger.info("optimal coverage estimate is set to {}".format(max_coverage))
            else:
                if self.config.bootstrap_trial > 0:
                    logger.info("determine margin using bootstrap method")
                    margin_list = []
                    for _ in tqdm.tqdm(range(self.config.bootstrap_trial)):
                        coverage_dummy = coverage_result.bootstrap_sampling()
                        margin = coverage_dummy.determine_margin(
                            self.config.acceptable_false_positive_rate
                        )
                        margin_list.append(margin)
                    margin = float(np.percentile(margin_list, self.config.bootstrap_percentile))
                    logger.info(margin_list)
                    logger.info("margin is set to {}".format(margin))
                else:
                    logger.info("determine margin without bootstrap method")
                    margin = coverage_result.determine_margin(
                        self.config.acceptable_false_positive_rate
                    )

                if not np.isfinite(margin) and self.config.ignore_useless_traj:
                    message = "margin value {} is invalid. retrun from active_sampling".format(
                        margin
                    )
                    logger.info(message)
                    return
                margins = [margin]

        # update library
        self.library.predictors.append(predictor)
        self.library.margins = margins
        self.library.coverage_results.append(coverage_result)

        # TODO: margins_history should not be Optional in the first place
        assert self.library._margins_history is not None
        self.library._margins_history.append(copy.deepcopy(margins))

        self.library.dump(self.project_path)

        coverage = self.library.measure_coverage(self.problems_validation)
        logger.info("current library's coverage estimate: {}".format(coverage))

    @property
    def difficult_iter_threshold(self) -> float:
        maxiter = self.library.solver_config.n_max_call
        difficult_iter_threshold = maxiter * self.config.difficult_threshold_factor
        return difficult_iter_threshold

    def learn_predictor(
        self,
        init_solution: Trajectory,
        project_path: Path,
        problems: List[ProblemT],
    ) -> Union[IterationPredictorWithEncoder, IterationPredictor]:
        pp = project_path

        logger.info("start generating dataset")
        # create dataset. Dataset creation by a large problem set often causes
        # memory error. To avoid this, we split the batch and process separately
        # if len(problems) > n_tau.
        init_solutions = [init_solution] * self.config.n_problem

        n_tau = 1000  # TODO: should be adaptive according to the data size
        partial_problems_list = [problems[i : i + n_tau] for i in range(0, len(problems), n_tau)]

        dataset = None
        for partial_problems in partial_problems_list:
            resultss_partial = self.solver.solve_batch(partial_problems, init_solutions)
            dataset_partial = IterationPredictorDataset.construct_from_tasks_and_resultss(
                init_solution,
                partial_problems,
                resultss_partial,
                self.solver_config,
                self.library.ae_model_shared,
            )
            # TODO: why don't you just use sum() method??
            # somehow error occurs: TypeError: unsupported operand type(s) for +: 'int' and 'IterationPredictorDataset'
            # I dont have time to fix this
            if dataset is None:
                dataset = dataset_partial
            else:
                dataset.add(dataset_partial)
        assert dataset is not None

        logger.info("start training model")
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

        model.initial_solution = init_solution
        tcache = TrainCache.from_model(model)

        def is_stoppable(tcache: TrainCache) -> bool:
            valid_losses = tcache.reduce_to_lossseq(tcache.validate_lossseq_table)
            n_step = len(valid_losses)
            idx_min = np.argmin(valid_losses)
            t_acceptable = 10
            no_improvement_for_long = bool((n_step - idx_min) > t_acceptable)
            return no_improvement_for_long

        train(pp, tcache, dataset, self.config.train_config, is_stoppable=is_stoppable)
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
                prob.invalidate_gridsdf()

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
    ) -> Optional[Trajectory]:
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
        return best_cand

    def _determine_init_solution(self) -> Trajectory:
        n_repeat_budget = 2
        for i_repeat in range(n_repeat_budget):
            logger.info("sample solution candidates ({}-th repeat)".format(i_repeat))
            problem_pool = self.pool_single
            solution_candidates = self._sample_solution_canidates(
                self.config.n_solution_candidate, problem_pool
            )
            assert (
                self.library._candidates_history is not None
            )  # FIXME: this never be None so this shouldnt be Optional
            self.library._candidates_history.append(solution_candidates)

            logger.info("sample difficult problems")
            if self.at_first_iteration():
                difficult_problems = [
                    next(problem_pool) for _ in range(self.config.n_difficult_problem)
                ]
            else:
                difficult_problems, _ = self._sample_difficult_problems(
                    self.config.n_difficult_problem, problem_pool
                )
            best_solution = self._select_solution_candidates(
                solution_candidates, difficult_problems
            )
            if best_solution is not None:
                logger.info("found best solution")
                return best_solution
        raise RuntimeError("consumed all repeat budget")
