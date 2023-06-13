import copy
import json
import logging
import pickle
import re
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Generic, List, Optional, Tuple, Type

import numpy as np
import threadpoolctl
import torch
import tqdm
from cmaes import CMA
from mohou.trainer import TrainCache, TrainConfig, train
from rpbench.interface import AbstractTaskSolver
from skmp.solver.interface import AbstractScratchSolver, ConfigT, ResultT
from skmp.trajectory import Trajectory

from hifuku.coverage import CoverageResult
from hifuku.datagen import (
    BatchProblemSampler,
    BatchProblemSolver,
    DistributeBatchProblemSampler,
    DistributedBatchProblemSolver,
    MultiProcessBatchProblemSampler,
    MultiProcessBatchProblemSolver,
)
from hifuku.neuralnet import (
    AutoEncoderBase,
    IterationPredictor,
    IterationPredictorConfig,
    IterationPredictorDataset,
)
from hifuku.pool import ProblemPool, ProblemT, TrivialProblemPool
from hifuku.types import get_clamped_iter
from hifuku.utils import num_torch_thread

logger = logging.getLogger(__name__)


def determine_margins(
    predictors: List[IterationPredictor],
    coverage_results: List[CoverageResult],
    threshold: float,
    target_fp_rate: float,
    cma_sigma: float,
    margins_guess: Optional[np.ndarray] = None,
) -> List[float]:
    def compute_coverage_and_fp(margins: np.ndarray) -> Tuple[float, float]:
        est_arr_list, real_arr_list = [], []
        for coverage_result, margin in zip(coverage_results, margins):
            est_arr_list.append(coverage_result.values_estimation + margin)
            real_arr_list.append(coverage_result.values_ground_truth)
        est_mat = np.array(est_arr_list)
        real_mat = np.array(real_arr_list)

        # find element indices which have smallest est value
        est_arr = np.min(est_mat, axis=0)
        element_indices = np.argmin(est_mat, axis=0)
        real_arr = np.array([real_mat[idx, i] for i, idx in enumerate(element_indices)])

        # compute (true + false) positive values
        est_bool_arr = est_arr < threshold
        real_bool_arr = real_arr < threshold

        n_total = len(est_arr)
        coverage_est = sum(est_bool_arr) / n_total

        # compute false positve rate
        fp_rate = sum(np.logical_and(est_bool_arr, ~real_bool_arr)) / sum(est_bool_arr)

        return coverage_est, fp_rate

    if margins_guess is None:
        n_pred = len(predictors)
        margins_guess = np.zeros(n_pred)
    optimizer = CMA(mean=margins_guess, sigma=cma_sigma)

    for generation in tqdm.tqdm(range(1000)):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            coverage_est, fp_rate = compute_coverage_and_fp(x)
            J = -coverage_est + 1000.0 * max(fp_rate - target_fp_rate, 0) ** 2
            solutions.append((x, J))
        optimizer.tell(solutions)
    logger.info("[cma result] coverage: {}, fp: {}".format(coverage_est, fp_rate))
    return list(x)


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
    ae_model: AutoEncoderBase
    predictors: List[IterationPredictor]
    margins: List[float]
    coverage_results: List[Optional[CoverageResult]]
    solvable_threshold_factor: float
    uuidval: str
    meta_data: Dict
    limit_thread: bool = False

    def __post_init__(self):
        assert self.ae_model.trained

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
        ae_model: AutoEncoderBase,
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
        )

    def put_on_device(self, device: torch.device):
        self.ae_model.put_on_device(device)
        for pred in self.predictors:
            pred.put_on_device(device)

    @property
    def device(self) -> torch.device:
        return self.ae_model.get_device()

    def _infer_iteration_num(self, task: ProblemT) -> np.ndarray:
        """
        itervals_arr: R^{n_solution, n_desc_inner}
        """
        assert len(self.predictors) > 0

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

        encoded: torch.Tensor = self.ae_model.encode(mesh)
        encoded_repeated = encoded.repeat(n_batch, 1)

        itervals_list = []
        for pred, margin in zip(self.predictors, self.margins):
            # margin is for correcting the overestimated inference
            itervals, _ = pred.forward((encoded_repeated, desc))
            itervals = itervals.squeeze(dim=0)
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
        init_solution_est_list = []
        for task in tqdm.tqdm(tasks):
            assert task.n_inner_task == 1
            infer_res = self.infer(task)[0]
            iterval_est_list.append(infer_res.nit)
            init_solution_est_list.append(infer_res.init_solution)

        logger.info("**compute real values")

        # When pickling-and-depickling, the procedure takes up much more memory than
        # pickled object size. I'm not sure but according to https://stackoverflow.com/a/38971446/7624196
        # the RAM usage could be two times bigger than the serialized-object size.
        # Thus, in the following, we first measure the pickle size, and splits the problem set
        # and then send the multiple chunk of problems sequentially.

        # TODO: ideally, this splitting and sequential-sending procedure should be
        # implemented inside http-based datagen class, as that problem happens only
        # sending serialized problems to server.
        max_ram_usage = 16 * 10**9
        serialize_ram_size_each = len(pickle.dumps(tasks[0])) * 2
        max_size = int(max_ram_usage // serialize_ram_size_each)
        indices = range(len(tasks))
        indices_list = np.array_split(indices, np.ceil(len(tasks) / max_size))

        results = []
        for indices_part in indices_list:
            tasks_part = [tasks[i] for i in indices_part]
            init_solution_est_list_part = [init_solution_est_list[i] for i in indices_part]
            results_part = solver.solve_batch(tasks_part, init_solution_est_list_part)
            results.extend(results_part)

        self.solver_config.n_max_call
        iterval_real_list = [get_clamped_iter(r[0], self.solver_config) for r in results]

        success_iter = self.success_iter_threshold()
        coverage_result = CoverageResult(
            np.array(iterval_real_list), np.array(iterval_est_list), success_iter
        )
        logger.info(coverage_result)
        return coverage_result

    def measure_coverage(self, tasks: List[ProblemT]) -> float:
        threshold = self.success_iter_threshold()
        count = 0
        for task in tasks:
            assert task.n_inner_task == 1
            infer_res = self.infer(task)[0]
            if infer_res.nit < threshold:
                count += 1
        return count / float(len(tasks))

    def dump(self, base_path: Path) -> None:
        cpu_device = torch.device("cpu")
        copied = copy.deepcopy(self)

        copied.ae_model.put_on_device(cpu_device)
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
        for path in base_path.iterdir():
            logger.debug("load from path: {}".format(path))
            m = re.match(r"Library-(\w+)-(\w+).pkl", path.name)
            if m is not None and m[1] == task_type.__name__:
                logger.info("library found at {}".format(path))
                with path.open(mode="rb") as f:
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
                ae_model=self.ae_model,
                predictors=[predictor],
                margins=[margin],
                coverage_results=[None],
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
            ae_model=singleton.ae_model,
            predictors=predictors,
            margins=margins,
            coverage_results=[None],
            solvable_threshold_factor=singleton.solvable_threshold_factor,
            uuidval="dummy",
            meta_data={},
        )
        return library


@dataclass
class LibraryBasedSolver(
    AbstractTaskSolver[ProblemT, ConfigT, ResultT], Generic[ProblemT, ConfigT, ResultT]
):
    library: SolutionLibrary[ProblemT, ConfigT, ResultT]
    solver: AbstractScratchSolver[ConfigT, ResultT]
    task: Optional[ProblemT]
    previous_false_positive: Optional[bool]

    @classmethod
    def init(
        cls, library: SolutionLibrary[ProblemT, ConfigT, ResultT]
    ) -> "LibraryBasedSolver[ProblemT, ConfigT, ResultT]":
        solver = library.solver_type.init(library.solver_config)
        return cls(library, solver, None, None)

    def setup(self, task: ProblemT) -> None:
        assert task.n_inner_task == 1
        p = task.export_problems()[0]
        self.solver.setup(p)
        self.task = task

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
            return result_type.abnormal(time.time() - ts)
        solver_result = self.solver.solve(inference_result.init_solution)
        solver_result.time_elapsed = time.time() - ts

        self.previous_false_positive = solver_result.traj is None
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
    sample_from_difficult_region: bool = True
    ignore_useless_traj: bool = True
    iterpred_model_config: Optional[Dict] = None
    bootstrap_trial: int = 0
    bootstrap_percentile: float = 95.0


@dataclass
class _SolutionLibrarySampler(Generic[ProblemT, ConfigT, ResultT], ABC):
    problem_type: Type[ProblemT]
    library: SolutionLibrary[ProblemT, ConfigT, ResultT]
    config: LibrarySamplerConfig
    pool_single: ProblemPool[ProblemT]
    pool_multiple: ProblemPool[ProblemT]
    problems_validation: List[ProblemT]
    solver: BatchProblemSolver
    sampler: BatchProblemSampler
    test_false_positive_rate: bool
    project_path: Path

    @property
    def solver_type(self) -> Type[AbstractScratchSolver[ConfigT, ResultT]]:
        return self.library.solver_type

    @property
    def solver_config(self) -> ConfigT:
        return self.library.solver_config

    def __post_init__(self):
        self.reset_pool()

    def reset_pool(self) -> None:
        logger.info("resetting pool")
        self.pool_single.reset()
        self.pool_multiple.reset()

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
    ) -> "_SolutionLibrarySampler[ProblemT, ConfigT, ResultT]":
        """
        use will be used only if either of solver and sampler is not set
        """
        meta_data = asdict(config)
        library = SolutionLibrary.initialize(
            problem_type,
            solver_t,
            solver_config,
            ae_model,
            config.solvable_threshold_factor,
            meta_data,
        )

        # setup solver and sampler
        if solver is None:
            solver = (
                DistributedBatchProblemSolver(solver_t, solver_config)
                if use_distributed
                else MultiProcessBatchProblemSolver(solver_t, solver_config)
            )
        assert solver.solver_t == solver_t
        assert solver.config == solver_config
        if sampler is None:
            sampler = (
                DistributeBatchProblemSampler()
                if use_distributed
                else MultiProcessBatchProblemSampler()
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
        parent_path = Path("/tmp/hifuku-validatino-set")
        parent_path.mkdir(exist_ok=True)
        validation_cache_path = parent_path / "{}-validation_set.pkl".format(problem_type.__name__)

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
                    10000, TrivialProblemPool(problem_type, 1).as_predicated()
                )
                with validation_cache_path.open(mode="wb") as f:
                    pickle.dump(problems_validation, f)
                logger.info(
                    "validation set with {} elements is created".format(len(problems_validation))
                )
        assert len(problems_validation) > 0

        for prob in problems_validation:
            assert prob.n_inner_task == 1
            logger.info(
                "validation set with {} elements is created".format(len(problems_validation))
            )

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
            test_false_positive_rate,
            project_path,
        )

    def _determine_init_solution_init(self) -> Trajectory:
        logger.info("start determine init solution using standard problem")
        task = self.problem_type.sample(1, standard=True)

        res = task.solve_default()[0]
        assert res.traj is not None
        return res.traj

    def _generate_problem_samples_init(self) -> List[ProblemT]:
        predicated_pool = self.pool_multiple.as_predicated()
        problems = self.sampler.sample_batch(self.config.n_problem, predicated_pool)
        return problems

    @abstractmethod
    def _determine_init_solution(self) -> Trajectory:
        ...

    @abstractmethod
    def _generate_problem_samples(self) -> List[ProblemT]:
        ...

    def step_active_sampling(self) -> None:
        logger.info("active sampling step")
        self.reset_pool()

        is_initialized = len(self.library.predictors) > 0
        if is_initialized:
            init_solution = self._determine_init_solution()
            problems = self._generate_problem_samples()
        else:
            init_solution = self._determine_init_solution_init()
            problems = self._generate_problem_samples_init()
        predictor = self.learn_predictor(init_solution, self.project_path, problems)

        logger.info("start measuring coverage")
        singleton_library = SolutionLibrary(
            task_type=self.problem_type,
            solver_type=self.solver_type,
            solver_config=self.solver_config,
            ae_model=self.library.ae_model,
            predictors=[predictor],
            margins=[0.0],
            coverage_results=[None],
            solvable_threshold_factor=self.config.solvable_threshold_factor,
            uuidval="dummy",
            meta_data={},
        )
        coverage_result = singleton_library.measure_full_coverage(
            self.problems_validation, self.solver
        )
        logger.info(coverage_result)

        with open("/tmp/hifuku_coverage_debug.pkl", "wb") as f:
            pickle.dump(coverage_result, f)

        if len(self.library.predictors) > 0:
            # determine std
            cma_std = self.solver_config.n_max_call * 0.5

            predictors_new = self.library.predictors + [predictor]
            coverages_new = self.library.coverage_results + [coverage_result]
            margins = determine_margins(
                predictors_new,
                coverages_new,
                self.solver_config.n_max_call,
                self.config.acceptable_false_positive_rate,
                cma_std,
            )
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
            margins = [margin]

        # update library
        self.library.predictors.append(predictor)
        self.library.margins = margins
        self.library.coverage_results.append(coverage_result)
        self.library.dump(self.project_path)

        coverage = self.library.measure_coverage(self.problems_validation)
        logger.info("current library's coverage estimate: {}".format(coverage))

        # # determine margin using bootstrap method

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
    ) -> IterationPredictor:
        pp = project_path
        cache_dir_base = pp / "dataset_cache"
        cache_dir_base.mkdir(exist_ok=True, parents=True)

        cache_dir_name = "{}-{}".format(self.problem_type.__name__, str(uuid.uuid4())[-8:])
        cache_dir_path = cache_dir_base / cache_dir_name
        assert not cache_dir_path.exists()
        cache_dir_path.mkdir()

        logger.info("start generating dataset")

        # create dataset. Dataset creation by a large problem set often causes
        # memory error. To avoid this, we split the batch and process separately
        # if len(problems) > n_tau.
        init_solutions = [init_solution] * self.config.n_problem

        n_tau = 2000  # TODO: should be adaptive according to the data size
        partial_problems_list = [problems[i : i + n_tau] for i in range(0, len(problems), n_tau)]
        for partial_problems in partial_problems_list:
            self.solver.create_dataset(
                partial_problems, init_solutions, cache_dir_path, n_process=None
            )

        dataset = IterationPredictorDataset.load(cache_dir_path, self.library.ae_model)
        shutil.rmtree(cache_dir_path)

        logger.info("start training model")

        # determine 1dim tensor dimension by temp creation of a problem
        # TODO: should I implement this as a method?
        problem = self.problem_type.sample(1, standard=True)
        table = problem.export_table()
        vector_desc = table.get_vector_descs()[0]
        n_dim_vector_description = vector_desc.shape[0]

        # train
        if self.config.iterpred_model_config is not None:
            model_conf = IterationPredictorConfig(
                n_dim_vector_description,
                self.library.ae_model.n_bottleneck,
                **self.config.iterpred_model_config
            )
        else:
            model_conf = IterationPredictorConfig(
                n_dim_vector_description, self.library.ae_model.n_bottleneck
            )

        model = IterationPredictor(model_conf)
        model.initial_solution = init_solution
        tcache = TrainCache.from_model(model)
        train(pp, tcache, dataset, self.config.train_config)
        return model

    def _sample_solution_canidates(
        self,
        n_sample: int,
        problem_pool: ProblemPool[ProblemT],
    ) -> List[Trajectory]:

        assert problem_pool.n_problem_inner == 1

        if self.config.sample_from_difficult_region:
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
        n_batch = n_sample * 20
        n_batch_little_difficult = int(n_batch * 0.8)
        n_batch_difficult = n_batch - n_batch_little_difficult

        feasible_solutions: List[Trajectory] = []
        while True:
            logger.info("{} sample batch".format(prefix))
            problems1 = self.sampler.sample_batch(
                n_batch_little_difficult, predicated_pool_bit_difficult
            )
            problems2 = self.sampler.sample_batch(n_batch_difficult, predicated_pool_difficult)
            problems = problems1 + problems2

            logger.info("{} solve batch".format(prefix))
            resultss = self.solver.solve_batch(problems, [None] * n_batch)

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
    ) -> Trajectory:
        logger.info("select single solution out of {} candidates".format(len(candidates)))

        score_list = []

        for i_cand, candidate in enumerate(candidates):
            solution_guesses = [candidate] * len(problems)
            # results = self.solver.solve_batch(problems, solution_guesses)
            # TODO: make flatten problem and use distributed
            solver = MultiProcessBatchProblemSolver[ConfigT, ResultT](
                self.solver_type, self.solver_config
            )  # distribute here is really slow
            results = solver.solve_batch(problems, solution_guesses)
            # consider all problems has n_inner_problem = 1
            iterval_real_list = [get_clamped_iter(r[0], self.solver_config) for r in results]
            score = -sum(iterval_real_list)  # must be nagative
            logger.debug("*score of solution candidate {}: {}".format(i_cand, score))
            score_list.append(score)

        best_idx = np.argmax(score_list)
        best_solution = candidates[best_idx]
        logger.debug("best score: {}".format(score_list[best_idx]))
        return best_solution


class SimpleSolutionLibrarySampler(_SolutionLibrarySampler[ProblemT, ConfigT, ResultT]):
    def _generate_problem_samples(self) -> List[ProblemT]:
        return self._generate_problem_samples_init()

    def _determine_init_solution(self) -> Trajectory:
        logger.info("sample solution candidates")
        problem_pool = self.pool_single
        solution_candidates = self._sample_solution_canidates(
            self.config.n_solution_candidate, problem_pool
        )

        logger.info("sample difficult problems")
        difficult_problems, _ = self._sample_difficult_problems(
            self.config.n_difficult_problem, problem_pool
        )
        best_solution = self._select_solution_candidates(solution_candidates, difficult_problems)
        return best_solution
