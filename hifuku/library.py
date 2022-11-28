import copy
import logging
import multiprocessing
import os
import pickle
import random
import re
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, List, Optional, Tuple, Type

import numpy as np
import pyclustering.cluster.xmeans as xmeans
import threadpoolctl
import torch
import tqdm
from mohou.trainer import TrainCache, TrainConfig, train

from hifuku.classifier import SVM, SVMDataset
from hifuku.datagen import (
    BatchProblemSampler,
    BatchProblemSolver,
    DistributeBatchProblemSampler,
    DistributedBatchProblemSolver,
    MultiProcessBatchProblemSampler,
    MultiProcessBatchProblemSolver,
    split_number,
)
from hifuku.llazy.dataset import LazyDecomplessDataset
from hifuku.margin import CoverageResult
from hifuku.neuralnet import (
    IterationPredictor,
    IterationPredictorConfig,
    IterationPredictorDataset,
    VoxelAutoEncoder,
)
from hifuku.pool import (
    FixedProblemPool,
    IteratorProblemPool,
    SimpleFixedProblemPool,
    SimpleIteratorProblemPool,
    TrivialIteratorPool,
)
from hifuku.types import ProblemT, RawData
from hifuku.utils import num_torch_thread

logger = logging.getLogger(__name__)


@dataclass
class SolutionLibrary(Generic[ProblemT]):
    problem_type: Type[ProblemT]
    ae_model: VoxelAutoEncoder
    predictors: List[IterationPredictor]
    margins: List[float]
    coverage_results: List[Optional[CoverageResult]]
    solvable_threshold_factor: float
    uuidval: str

    def __post_init__(self):
        assert self.ae_model.loss_called

    @dataclass
    class InferenceResult:
        nit: float
        idx: int  # index of selected solution in the library
        init_solution: np.ndarray

    @classmethod
    def initialize(
        cls,
        problem_type: Type[ProblemT],
        ae_model: VoxelAutoEncoder,
        solvable_threshold_factor: float,
    ) -> "SolutionLibrary[ProblemT]":
        uuidval = str(uuid.uuid4())[-8:]
        return cls(problem_type, ae_model, [], [], [], solvable_threshold_factor, uuidval)

    def _put_on_device(self, device: torch.device):
        self.ae_model.put_on_device(device)
        for pred in self.predictors:
            pred.put_on_device(device)

    @property
    def device(self) -> torch.device:
        return self.ae_model.device

    def _infer_iteration_num(self, problem: ProblemT) -> np.ndarray:
        """
        itervals_arr: R^{n_solution, n_problem_inner}
        """
        assert len(self.predictors) > 0

        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            # limiting numpy thread seems to make stable. but not sure why..
            mesh_np = np.expand_dims(problem.get_mesh(), axis=(0, 1))
            desc_np = np.array(problem.get_descriptions())

        with num_torch_thread(1):
            # float() must be run in single (cpp-layer) thread
            # see https://github.com/pytorch/pytorch/issues/89693
            mesh = torch.from_numpy(mesh_np)
            mesh = mesh.float().to(self.device)
            desc = torch.from_numpy(desc_np)
            desc = desc.float().to(self.device)

        n_batch, _ = desc_np.shape

        encoded: torch.Tensor = self.ae_model.encoder(mesh)
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

    def infer(self, problem: ProblemT) -> List[InferenceResult]:
        # itervals_aar: R^{n_problem_inner, n_elem_in_lib}
        itervals_arr = self._infer_iteration_num(problem)

        # nits_min: R^{n_problem_inner}
        nits_min = np.min(itervals_arr, axis=0)

        # indices_min: R^{n_problem_inner}
        indices_min = np.argmin(itervals_arr, axis=0)

        result_list = []
        for nit, idx in zip(nits_min, indices_min):
            init_solution = self.predictors[idx].initial_solution
            assert init_solution is not None
            res = self.InferenceResult(nit, idx, init_solution)
            result_list.append(res)
        return result_list

    def success_iter_threshold(self) -> float:
        config = self.problem_type.get_solver_config()
        threshold = config.maxiter * self.solvable_threshold_factor
        return threshold

    def measure_full_coverage(
        self, problem_pool: FixedProblemPool[ProblemT], solver: BatchProblemSolver
    ) -> CoverageResult:
        logger.info("**compute est values")
        iterval_est_list = []
        init_solution_est_list = []
        for problem in tqdm.tqdm(problem_pool):
            assert problem.n_problem() == 1
            infer_res = self.infer(problem)[0]
            iterval_est_list.append(infer_res.nit)
            init_solution_est_list.append(infer_res.init_solution)

        logger.info("**compute real values")
        problems = [p for p in problem_pool]
        results = solver.solve_batch(problems, init_solution_est_list)

        maxiter = self.problem_type.get_solver_config().maxiter
        iterval_real_list = [(maxiter if not r[0].success else r[0].nit) for r in results]

        success_iter = self.success_iter_threshold()
        coverage_result = CoverageResult(
            np.array(iterval_real_list), np.array(iterval_est_list), success_iter
        )
        logger.info(coverage_result)
        return coverage_result

    def measure_coverage(self, problem_pool: FixedProblemPool[ProblemT]) -> float:
        threshold = self.success_iter_threshold()
        count = 0
        for problem in problem_pool:
            assert problem.n_problem() == 1
            infer_res = self.infer(problem)[0]
            if infer_res.nit < threshold:
                count += 1
        return count / float(len(problem_pool))

    def add(
        self,
        predictor: IterationPredictor,
        margin: float,
        coverage_reuslt: Optional[CoverageResult],
    ):
        self.predictors.append(predictor)
        self.margins.append(margin)
        self.coverage_results.append(coverage_reuslt)

    def dump(self, base_path: Path) -> None:
        cpu_device = torch.device("cpu")
        copied = copy.deepcopy(self)

        copied.ae_model.put_on_device(cpu_device)
        for pred in copied.predictors:
            pred.put_on_device(cpu_device)

        name = "Library-{}-{}.pkl".format(self.problem_type.__name__, self.uuidval)
        file_path = base_path / name
        with file_path.open(mode="wb") as f:
            pickle.dump(copied, f)
        logger.info("dumped library to {}".format(file_path))

    @classmethod
    def load(
        cls,
        base_path: Path,
        problem_type: Type[ProblemT],
        device: Optional[torch.device] = None,
    ) -> List["SolutionLibrary[ProblemT]"]:
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        libraries = []
        for path in base_path.iterdir():
            m = re.match(r"Library-(\w+)-(\w+).pkl", path.name)
            if m is not None and m[1] == problem_type.__name__:
                logger.info("library found at {}".format(path))
                with path.open(mode="rb") as f:
                    lib: "SolutionLibrary[ProblemT]" = pickle.load(f)
                    assert lib.device == torch.device("cpu")
                    lib._put_on_device(device)
                    libraries.append(lib)
        return libraries  # type: ignore


@dataclass
class LargestDifficultClusterPredicate(Generic[ProblemT]):
    library: SolutionLibrary[ProblemT]
    svm: SVM
    accept_proba_threshold: float

    def __post_init__(self):
        assert self.library.device == torch.device("cpu")

    @classmethod
    def create(
        cls,
        library: SolutionLibrary[ProblemT],
        difficult_problems: List[ProblemT],
        ambient_problems: List[ProblemT],
        accept_proba_threshold: float = 0.4,
    ) -> "LargestDifficultClusterPredicate[ProblemT]":
        """
        difficult problems for detect the largest cluster
        ambient_problems + difficult_problems for fit the clf
        """

        # sanity check (only first element)c:
        assert difficult_problems[0].n_problem() == 1
        assert ambient_problems[0].n_problem() == 1

        # lirary should be put on cpu
        cpu_device = torch.device("cpu")
        if library.device != cpu_device:
            logger.debug("library is on gpu. copy and put the library on cpu")
            library = copy.deepcopy(library)
            library._put_on_device(cpu_device)

        difficult_iters_list = [
            library._infer_iteration_num(p).flatten() for p in tqdm.tqdm(difficult_problems)
        ]
        easy_iters_list = [
            library._infer_iteration_num(p).flatten() for p in tqdm.tqdm(ambient_problems)
        ]

        initializer = xmeans.kmeans_plusplus_initializer(
            data=difficult_iters_list, amount_centers=2
        )
        initial_centers = initializer.initialize()
        xm = xmeans.xmeans(data=difficult_iters_list, initial_centers=initial_centers)
        xm.process()
        clusters = xm.get_clusters()
        larget_cluster_indices: np.ndarray = sorted(clusters, key=lambda c: len(c))[-1]  # type: ignore
        logger.info("{} clusters with {} elements".format(len(clusters), [len(c) for c in clusters]))  # type: ignore

        X = difficult_iters_list + easy_iters_list
        Y = np.zeros(len(X), dtype=bool)
        Y[larget_cluster_indices] = True
        dataset = SVMDataset.from_xy(X, Y)
        svm = SVM.from_dataset(dataset)
        return cls(library, svm, accept_proba_threshold)

    def __call__(self, problem: ProblemT) -> bool:
        assert problem.n_problem() == 1
        iters = self.library._infer_iteration_num(problem).flatten()
        proba = self.svm.predict_proba(iters)
        return proba > self.accept_proba_threshold

    @staticmethod
    def task(
        library: SolutionLibrary[ProblemT],
        svm: SVM,
        n_sample: int,
        pool: IteratorProblemPool[ProblemT],
        accept_threshold: float,
        ambient_rate: float,
        show_progress_bar: bool,
        n_thread: int,
        cache_path: Path,
    ) -> None:
        def predicate(problem: ProblemT) -> bool:
            assert problem.n_problem() == 1
            iters = library._infer_iteration_num(problem).flatten()
            proba = svm.predict_proba(iters)
            return proba > accept_threshold

        predicated_pool = pool.make_predicated(predicate, 40)

        # set random seed
        unique_id = (uuid.getnode() + os.getpid()) % (2**32 - 1)
        np.random.seed(unique_id)
        logger.debug("random seed set to {}".format(unique_id))

        logger.debug("start sampling using clf")
        problems: List[ProblemT] = []
        n_ambient = int(n_sample * ambient_rate)
        n_sample_focus = n_sample - n_ambient

        with num_torch_thread(n_thread):
            with tqdm.tqdm(
                total=n_sample_focus, smoothing=0.0, disable=not show_progress_bar
            ) as pbar:
                while len(problems) < n_sample_focus:
                    problem = next(predicated_pool)
                    if problem is not None:
                        problems.append(problem)
                        pbar.update(1)

        for _ in range(n_ambient):
            problems.append(next(pool))

        random.seed(0)
        random.shuffle(problems)  # noqa

        ts = time.time()
        file_path = cache_path / str(uuid.uuid4())
        with file_path.open(mode="wb") as f:
            pickle.dump(problems, f)
        logger.debug("time to dump {}".format(time.time() - ts))

    def sample(
        self,
        n_sample: int,
        pool: IteratorProblemPool[ProblemT],
        accept_threshold: float = 0.4,
        ambient_rate: float = 0.2,
        n_process: Optional[int] = None,
    ) -> List[ProblemT]:

        cpu_count = os.cpu_count()
        assert cpu_count is not None
        n_physical_cpu = int(0.5 * cpu_count)

        if n_process is None:
            good_thread_num = 2  # from my experience
            n_process = n_physical_cpu // good_thread_num
        assert n_sample > n_process * 5  # this is random. i don't have time

        with tempfile.TemporaryDirectory() as td:
            # https://github.com/pytorch/pytorch/issues/89693
            ctx = multiprocessing.get_context(method="spawn")
            n_sample_list = split_number(n_sample, n_process)
            process_list = []

            td_path = Path(td)
            n_thread = n_physical_cpu // n_process
            for idx_process, n_sample_part in enumerate(n_sample_list):
                show_progress = idx_process == 0
                args = (
                    self.library,
                    self.svm,
                    n_sample_part,
                    pool,
                    accept_threshold,
                    ambient_rate,
                    show_progress,
                    n_thread,
                    td_path,
                )
                p = ctx.Process(target=self.task, args=args)
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()

            ts = time.time()
            problems_sampled = []
            for file_path in td_path.iterdir():
                with file_path.open(mode="rb") as f:
                    problems_sampled.extend(pickle.load(f))
            print("time to load {}".format(time.time() - ts))
        return problems_sampled


@dataclass
class LibrarySamplerConfig:
    n_problem: int
    n_problem_inner: int
    train_config: TrainConfig
    n_solution_candidate: int = 10
    n_difficult_problem: int = 100
    solvable_threshold_factor: float = 0.8
    difficult_threshold_factor: float = 0.8  # should equal to solvable_threshold_factor
    acceptable_false_positive_rate: float = 0.05


@dataclass
class _SolutionLibrarySampler(Generic[ProblemT], ABC):
    problem_type: Type[ProblemT]
    library: SolutionLibrary[ProblemT]
    config: LibrarySamplerConfig
    pool_single: IteratorProblemPool[ProblemT]
    pool_multiple: IteratorProblemPool[ProblemT]
    pool_validation: FixedProblemPool[ProblemT]
    solver: BatchProblemSolver
    sampler: BatchProblemSampler

    @classmethod
    def initialize(
        cls,
        problem_type: Type[ProblemT],
        ae_model: VoxelAutoEncoder,
        config: LibrarySamplerConfig,
        pool_single: Optional[IteratorProblemPool[ProblemT]] = None,
        pool_multiple: Optional[IteratorProblemPool[ProblemT]] = None,
        pool_validation: Optional[FixedProblemPool[ProblemT]] = None,
        solver: Optional[BatchProblemSolver[ProblemT]] = None,
        sampler: Optional[BatchProblemSampler[ProblemT]] = None,
        use_distributed: bool = False,
    ) -> "_SolutionLibrarySampler[ProblemT]":
        """
        use will be used only if either of solver and sampler is not set
        """
        library = SolutionLibrary.initialize(
            problem_type, ae_model, config.solvable_threshold_factor
        )

        # setup pools
        if pool_single is None:
            logger.info("problem pool is not specified. use SimpleProblemPool")
            pool_single = SimpleIteratorProblemPool(problem_type, 1)
        assert pool_single.n_problem_inner == 1

        if pool_multiple is None:
            logger.info("problem pool is not specified. use SimpleProblemPool")
            # TODO: smelling! n_problem_inner should not be set here
            pool_multiple = SimpleIteratorProblemPool(problem_type, config.n_problem_inner)

        if pool_validation is None:
            pool_validation = SimpleFixedProblemPool.initialize(problem_type, 1000)
        assert pool_validation.n_problem_inner == 1

        # setup solver and sampler
        if solver is None:
            solver = (
                DistributedBatchProblemSolver()
                if use_distributed
                else MultiProcessBatchProblemSolver()
            )
        if sampler is None:
            sampler = (
                DistributeBatchProblemSampler()
                if use_distributed
                else MultiProcessBatchProblemSampler()
            )

        logger.info("library sampler config: {}".format(config))
        return cls(
            problem_type,
            library,
            config,
            pool_single,
            pool_multiple,
            pool_validation,
            solver,
            sampler,
        )

    def _determine_init_solution_init(self) -> np.ndarray:
        logger.info("start determine init solution using standard problem")
        init_solution = self.problem_type.get_default_init_solution()
        return init_solution

    def _generate_problem_samples_init(self) -> List[ProblemT]:
        predicated_pool = self.pool_multiple.as_predicated()
        problems = self.sampler.sample_batch(self.config.n_problem, predicated_pool)
        return problems

    @abstractmethod
    def _determine_init_solution(self) -> np.ndarray:
        ...

    @abstractmethod
    def _generate_problem_samples(self) -> List[ProblemT]:
        ...

    def step_active_sampling(
        self,
        project_path: Path,
    ) -> None:
        logger.info("active sampling step")

        is_initialized = len(self.library.predictors) > 0
        if is_initialized:
            init_solution = self._determine_init_solution()
            problems = self._generate_problem_samples()
        else:
            init_solution = self._determine_init_solution_init()
            problems = self._generate_problem_samples_init()
        predictor = self.learn_predictors(init_solution, project_path, problems)

        logger.info("start measuring coverage")
        singleton_library = SolutionLibrary(
            problem_type=self.problem_type,
            ae_model=self.library.ae_model,
            predictors=[predictor],
            margins=[0.0],
            coverage_results=[None],
            solvable_threshold_factor=self.config.solvable_threshold_factor,
            uuidval="dummy",
        )
        coverage_result = singleton_library.measure_full_coverage(self.pool_validation, self.solver)
        logger.info(coverage_result)
        margin = coverage_result.determine_margin(self.config.acceptable_false_positive_rate)

        logger.info("margin is set to {}".format(margin))
        self.library.add(predictor, margin, coverage_result)

        coverage = self.library.measure_coverage(self.pool_validation)
        logger.info("current library's coverage estimate: {}".format(coverage))

        self.library.dump(project_path)

    @property
    def difficult_iter_threshold(self) -> float:
        difficult_iter_threshold = (
            self.problem_type.get_solver_config().maxiter * self.config.difficult_threshold_factor
        )
        return difficult_iter_threshold

    def learn_predictors(
        self,
        init_solution: np.ndarray,
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

        # create dataset
        init_solutions = [init_solution] * self.config.n_problem
        self.solver.create_dataset(problems, init_solutions, cache_dir_path, n_process=None)

        dataset = IterationPredictorDataset.load(cache_dir_path, self.library.ae_model)
        raw_dataset = LazyDecomplessDataset.load(cache_dir_path, RawData, n_worker=-1)

        rawdata = raw_dataset.get_data(np.array([0]))[0]
        init_solution = rawdata.init_solution

        logger.info("start training model")
        model_conf = IterationPredictorConfig(12, self.library.ae_model.config.dim_bottleneck, 10)
        model = IterationPredictor(model_conf)
        model.initial_solution = init_solution
        tcache = TrainCache.from_model(model)
        train(pp, tcache, dataset, self.config.train_config)
        return model

    def _sample_solution_canidates(
        self,
        n_sample: int,
        problem_pool: IteratorProblemPool[ProblemT],
    ) -> List[np.ndarray]:
        difficult_iter_threshold = self.difficult_iter_threshold

        solution_candidates: List[np.ndarray] = []
        with tqdm.tqdm(total=n_sample) as pbar:
            while len(solution_candidates) < n_sample:
                problem = next(problem_pool)
                assert problem.n_problem() == 1
                infer_res = self.library.infer(problem)[0]
                iterval = infer_res.nit

                is_difficult = iterval > difficult_iter_threshold
                if is_difficult:
                    logger.debug("try solving a difficult problem")
                    assert problem.n_problem() == 1
                    try:
                        res = problem.solve()[0]
                    except self.problem_type.SamplingBasedInitialguessFail:
                        continue
                    if not res.success:
                        continue
                    if res is not None:  # seems feasible
                        assert res.success
                        solution_candidates.append(res.x)
                        logger.debug(
                            "solved difficult problem. current num => {}".format(
                                len(solution_candidates)
                            )
                        )
                        pbar.update(1)
        return solution_candidates

    def _sample_difficult_problems(
        self,
        n_sample: int,
        problem_pool: IteratorProblemPool[ProblemT],
    ) -> Tuple[List[ProblemT], List[ProblemT]]:

        difficult_problems: List[ProblemT] = []
        easy_problems: List[ProblemT] = []
        with tqdm.tqdm(total=n_sample) as pbar:
            while len(difficult_problems) < n_sample:
                logger.debug("try sampling difficutl problem...")
                problem = next(problem_pool)
                assert problem.n_problem() == 1
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
        self, candidates: List[np.ndarray], problems: List[ProblemT]
    ) -> np.ndarray:
        logger.info("compute scores")
        score_list = []
        maxiter = self.problem_type.get_solver_config().maxiter
        for candidate in candidates:
            solution_guesses = [candidate] * len(problems)
            # results = self.solver.solve_batch(problems, solution_guesses)
            # TODO: make flatten problem and use distributed
            solver = MultiProcessBatchProblemSolver[ProblemT]()  # distribute here is really slow
            results = solver.solve_batch(problems, solution_guesses)
            # consider all problems has n_inner_problem = 1
            iterval_real_list = [(maxiter if not r[0].success else r[0].nit) for r in results]
            score = -sum(iterval_real_list)  # must be nagative
            logger.debug("*score of solution cand: {}".format(score))
            score_list.append(score)

        best_idx = np.argmax(score_list)
        best_solution = candidates[best_idx]
        logger.debug("best score: {}".format(score_list[best_idx]))
        return best_solution


class SimpleSolutionLibrarySampler(_SolutionLibrarySampler[ProblemT]):
    def _generate_problem_samples(self) -> List[ProblemT]:
        return self._generate_problem_samples_init()

    def _determine_init_solution(self) -> np.ndarray:
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


class ClusterBasedSolutionLibrarySampler(_SolutionLibrarySampler[ProblemT]):
    predicate_cache: Optional[LargestDifficultClusterPredicate] = None

    def _generate_problem_samples(self) -> List[ProblemT]:
        assert self.predicate_cache is not None
        n_problem_half = int(self.config.n_problem * 0.5)
        predicated_pool = self.pool_multiple.make_predicated(
            self.predicate_cache, max_trial_factor=50
        )
        problems_in_clf = self.sampler.sample_batch(n_problem_half, predicated_pool)
        problems_ambient = self.sampler.sample_batch(
            n_problem_half, self.pool_multiple.as_predicated()
        )
        problems = problems_in_clf + problems_ambient
        return problems

    def _determine_init_solution(self) -> np.ndarray:
        logger.info("sample solution candidates")

        n_sample_difficult = 1000
        logger.info("sample difficult problem")
        difficult_problems, easy_problems = self._sample_difficult_problems(
            n_sample_difficult, self.pool_single
        )
        logger.debug(
            "n_difficult: {}, n_easy: {}".format(len(difficult_problems), len(easy_problems))
        )
        n_remainder = max(0, n_sample_difficult - len(easy_problems))
        if n_remainder > 0:
            logger.debug("additional easy {} problems sampling".format(n_remainder))
            additional = self.sampler.sample_batch(n_remainder, self.pool_single.as_predicated())
            easy_problems.extend(additional)
        easy_problems = easy_problems[:n_sample_difficult]

        predicate = LargestDifficultClusterPredicate.create(
            self.library, difficult_problems, easy_problems
        )
        self.predicate_cache = predicate

        predicated_pool = self.pool_single.make_predicated(predicate, max_trial_factor=50)
        n_problem_half = int(self.config.n_problem * 0.5)
        logger.info("sample in-clf problems")
        problems_in_clf = self.sampler.sample_batch(n_problem_half, predicated_pool)

        n_max_trial = 10
        trial_count = 0
        while True:
            trial_count += 1
            logger.debug("trial count increment to {}".format(trial_count))
            iter_pool = TrivialIteratorPool(problems_in_clf.__iter__())
            try:
                solution_candidates = self._sample_solution_canidates(
                    self.config.n_solution_candidate, iter_pool
                )
                break
            except StopIteration:
                if trial_count > n_max_trial:
                    assert False, "reached max trial"
                # if not enough, double the size
                logger.debug("iter pool size is not enough. do additional sampling")
                n_current_size = len(problems_in_clf)
                additional_problem_in_clf = self.sampler.sample_batch(
                    2 * n_current_size, predicated_pool
                )
                problems_in_clf.extend(additional_problem_in_clf)  # dobuled
        assert len(problems_in_clf) > self.config.n_difficult_problem

        problems_for_eval = problems_in_clf[: self.config.n_difficult_problem]
        best_solution = self._select_solution_candidates(solution_candidates, problems_for_eval)
        return best_solution
