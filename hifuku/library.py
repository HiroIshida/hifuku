import logging
import multiprocessing
import os
import pickle
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterable, Iterator, List, Optional, Sequence, Sized, Type

import numpy as np
import torch
import tqdm
from mohou.trainer import TrainCache, TrainConfig, train

from hifuku.datagen import DatasetGenerator
from hifuku.llazy.dataset import LazyDecomplessDataset
from hifuku.margin import CoverageResult
from hifuku.neuralnet import (
    IterationPredictor,
    IterationPredictorConfig,
    IterationPredictorDataset,
    VoxelAutoEncoder,
)
from hifuku.types import ProblemInterface, ProblemT, RawData

logger = logging.getLogger(__name__)


@dataclass
class ComputeRealItervalsArg:
    indices: np.ndarray
    problems: Sequence[ProblemInterface]
    init_solution: np.ndarray
    maxiter: int
    disable_tqdm: bool


def _compute_real_itervals(arg: ComputeRealItervalsArg, q: multiprocessing.Queue):
    with tqdm.tqdm(total=len(arg.problems), disable=arg.disable_tqdm) as pbar:
        for idx, problem in zip(arg.indices, arg.problems):
            assert problem.n_problem() == 1
            result = problem.solve(arg.init_solution)[0]
            iterval_real = result.nit if result.success else float(arg.maxiter)
            q.put((idx, iterval_real))
            pbar.update(1)


def compute_real_itervals(
    problems: Sequence[ProblemInterface],
    init_solution: np.ndarray,
    maxiter: int,
    n_process: Optional[int] = None,
) -> List[float]:
    if n_process is None:
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        n_process = int(0.5 * cpu_count)

    is_single_process = n_process == 1
    if is_single_process:
        itervals = []
        for problem in problems:
            result = problem.solve(init_solution)[0]
            iterval_real = result.nit if result.success else float(maxiter)
            itervals.append(iterval_real)
        return itervals
    else:
        indices = np.array(list(range(len(problems))))
        indices_list_per_worker = np.array_split(indices, n_process)

        q = multiprocessing.Queue()  # type: ignore
        indices_list_per_worker = np.array_split(indices, n_process)

        process_list = []
        for i, indices_part in enumerate(indices_list_per_worker):
            disable_tqdm = i > 0
            problems_part = [problems[idx] for idx in indices_part]
            arg = ComputeRealItervalsArg(
                indices_part, problems_part, init_solution, maxiter, disable_tqdm
            )
            p = multiprocessing.Process(target=_compute_real_itervals, args=(arg, q))
            p.start()
            process_list.append(p)

        idx_iterval_pairs = [q.get() for _ in range(len(problems))]
        idx_iterval_pairs_sorted = sorted(idx_iterval_pairs, key=lambda x: x[0])  # type: ignore
        _, itervals = zip(*idx_iterval_pairs_sorted)
        return list(itervals)


class ProblemPool(Iterable[ProblemT]):
    problem_type: Type[ProblemT]


class IteratorProblemPool(Iterator[ProblemT], ProblemPool[ProblemT]):
    pass


@dataclass
class SimpleProblemPool(IteratorProblemPool[ProblemT]):
    problem_type: Type[ProblemT]

    def __next__(self) -> ProblemT:
        return self.problem_type.sample(1)


class FixedProblemPool(Sized, ProblemPool[ProblemT]):
    @abstractmethod
    def __len__(self) -> int:
        ...


@dataclass
class SimpleFixedProblemPool(FixedProblemPool[ProblemT]):
    problem_type: Type[ProblemT]
    problem_list: List[ProblemT]

    @classmethod
    def initialize(cls, problem_type: Type[ProblemT], n_problem: int) -> "SimpleFixedProblemPool":
        problem_list = [problem_type.sample(1) for _ in range(n_problem)]
        return cls(problem_type, problem_list)

    def __len__(self) -> int:
        return len(self.problem_list)

    def __iter__(self) -> Iterator[ProblemT]:
        return self.problem_list.__iter__()


@dataclass
class SolutionLibrary(Generic[ProblemT]):
    problem_type: Type[ProblemT]
    ae_model: VoxelAutoEncoder
    predictors: List[IterationPredictor]
    margins: List[float]
    coverage_results: List[Optional[CoverageResult]]
    solvable_threshold_factor: float
    uuidval: str

    @classmethod
    def initialize(
        cls,
        problem_type: Type[ProblemT],
        ae_model: VoxelAutoEncoder,
        solvable_threshold_factor: float,
    ) -> "SolutionLibrary[ProblemT]":
        uuidval = str(uuid.uuid4())[-8:]
        return cls(problem_type, ae_model, [], [], [], solvable_threshold_factor, uuidval)

    @property
    def device(self) -> torch.device:
        return self.ae_model.device

    def _infer_iteration_num(self, problem: ProblemT) -> np.ndarray:
        """
        itervals_arr: R^{n_solution, n_problem_inner}
        """
        assert len(self.predictors) > 0

        mesh_np = np.expand_dims(problem.get_mesh(), axis=(0, 1))
        mesh = torch.from_numpy(mesh_np).float().to(self.device)

        desc_np = np.array(problem.get_descriptions())
        desc = torch.from_numpy(desc_np).float()
        desc = desc.to(self.device)
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

    def infer_iteration_num(self, problem: ProblemT) -> np.ndarray:
        """
        itervals_arr: R^{n_problem_inner}
        """
        itervals_arr = self._infer_iteration_num(problem)
        itervals_min = np.min(itervals_arr, axis=0)
        return itervals_min

    def success_iter_threshold(self) -> float:
        config = self.problem_type.get_solver_config()
        threshold = config.maxiter * self.solvable_threshold_factor
        return threshold

    def measure_coverage(self, problem_pool: FixedProblemPool[ProblemT]) -> float:
        threshold = self.success_iter_threshold()
        count = 0
        for problem in problem_pool:
            assert problem.n_problem() == 1
            iterval = self.infer_iteration_num(problem)[0].item()
            if iterval < threshold:
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
        name = "Library-{}-{}.pkl".format(self.problem_type.__name__, self.uuidval)
        file_path = base_path / name
        with file_path.open(mode="wb") as f:
            pickle.dump(self, f)
        logger.info("dumped library to {}".format(file_path))

    @classmethod
    def load(
        cls, base_path: Path, problem_type: Type[ProblemT]
    ) -> List["SolutionLibrary[ProblemT]"]:
        libraries = []
        for path in base_path.iterdir():
            m = re.match(r"Library-(\w+)-(\w+).pkl", path.name)
            if m is not None and m[1] == problem_type.__name__:
                logger.info("library found at {}".format(path))
                with path.open(mode="rb") as f:
                    libraries.append(pickle.load(f))
        return libraries  # type: ignore


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
class SolutionLibrarySampler(Generic[ProblemT], ABC):
    problem_type: Type[ProblemT]
    library: SolutionLibrary[ProblemT]
    dataset_gen: DatasetGenerator
    config: LibrarySamplerConfig
    validation_problem_pool: FixedProblemPool[ProblemT]
    coverage_result_list: List[CoverageResult]

    @classmethod
    def initialize(
        cls,
        problem_type: Type[ProblemT],
        ae_model: VoxelAutoEncoder,
        dataset_gen: DatasetGenerator,
        config: LibrarySamplerConfig,
        validation_problem_pool: FixedProblemPool[ProblemT],
    ) -> "SolutionLibrarySampler[ProblemT]":
        library = SolutionLibrary.initialize(
            problem_type, ae_model, config.solvable_threshold_factor
        )
        logger.info("library sampler config: {}".format(config))
        return cls(problem_type, library, dataset_gen, config, validation_problem_pool, [])

    def step_active_sampling(
        self, project_path: Path, problem_pool: Optional[IteratorProblemPool[ProblemT]] = None
    ):
        logger.info("active sampling step")

        if problem_pool is None:
            logger.info("problem pool is not specified. use SimpleProblemPool")
            problem_pool = SimpleProblemPool(self.problem_type)

        init_solution = self._determine_init_solution(problem_pool)
        predictor = self.learn_predictors(init_solution, project_path)

        coverage_result = self._compute_coverage(predictor, init_solution)
        margin = coverage_result.determine_margin(self.config.acceptable_false_positive_rate)

        logger.info("margin is set to {}".format(margin))
        self.coverage_result_list.append(coverage_result)
        self.library.add(predictor, margin, coverage_result)

        coverage = self.library.measure_coverage(self.validation_problem_pool)
        logger.info("current library's coverage estimate: {}".format(coverage))

    def _compute_coverage(
        self, predictor: IterationPredictor, init_solution: np.ndarray
    ) -> CoverageResult:

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

        maxiter = self.problem_type.get_solver_config().maxiter

        logger.info("**compute est values")
        iterval_est_list = []
        for problem in tqdm.tqdm(self.validation_problem_pool):
            assert problem.n_problem() == 1
            iterval_est = singleton_library.infer_iteration_num(problem)[0]
            iterval_est_list.append(iterval_est)

        logger.info("**compute real values")
        problems = [p for p in self.validation_problem_pool]
        iterval_real_list = compute_real_itervals(problems, init_solution, maxiter)

        success_iter_threshold = maxiter * self.config.difficult_threshold_factor
        coverage_result = CoverageResult(
            np.array(iterval_real_list), np.array(iterval_est_list), success_iter_threshold
        )
        logger.info(coverage_result)
        return coverage_result

    def learn_predictors(self, init_solution: np.ndarray, project_path: Path) -> IterationPredictor:
        pp = project_path
        cache_dir_base = pp / "dataset_cache"
        cache_dir_base.mkdir(exist_ok=True, parents=True)

        cache_dir_name = "{}-{}".format(self.problem_type.__name__, str(uuid.uuid4())[-8:])
        cache_dir_path = cache_dir_base / cache_dir_name
        assert not cache_dir_path.exists()
        cache_dir_path.mkdir()

        logger.info("start generating dataset")
        self.dataset_gen.generate(
            init_solution, self.config.n_problem, self.config.n_problem_inner, cache_dir_path
        )

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

    def _determine_init_solution(self, problem_pool: IteratorProblemPool[ProblemT]) -> np.ndarray:

        is_initialized = len(self.library.predictors) > 0
        if not is_initialized:
            logger.info("start determine init solution using standard problem")
            init_solution = self.problem_type.get_default_init_solution()
            return init_solution
        else:
            logger.info("start determine init solution len(lib) > 0")

            logger.info("sample solution candidates")
            solution_candidates = self._sample_solution_canidates(problem_pool)

            logger.info("sample difficult problems")
            difficult_problems = self._sample_difficult_problems(problem_pool)

            logger.info("compute scores")
            score_list = []
            for solution_guess in solution_candidates:
                score = 0.0
                for problem in difficult_problems:
                    res = problem.solve(solution_guess)[0]
                    score += int(res.success)
                score_list.append(score)
            best_solution = solution_candidates[np.argmax(score_list)]
            return best_solution

    def _sample_solution_canidates(
        self, problem_pool: IteratorProblemPool[ProblemT]
    ) -> List[np.ndarray]:
        maxiter = self.problem_type.get_solver_config().maxiter
        difficult_iter_threshold = maxiter * self.config.difficult_threshold_factor

        solution_candidates: List[np.ndarray] = []
        with tqdm.tqdm(total=self.config.n_solution_candidate) as pbar:
            while len(solution_candidates) < self.config.n_solution_candidate:
                problem = next(problem_pool)
                assert problem.n_problem() == 1
                iterval = self.library.infer_iteration_num(problem)[0]

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
        self, problem_pool: IteratorProblemPool[ProblemT]
    ) -> List[ProblemT]:

        maxiter = self.problem_type.get_solver_config().maxiter
        difficult_iter_threshold = maxiter * self.config.difficult_threshold_factor

        difficult_problems: List[ProblemT] = []
        with tqdm.tqdm(total=self.config.n_difficult_problem) as pbar:
            while len(difficult_problems) < self.config.n_difficult_problem:
                logger.debug("try sampling difficutl problem...")
                problem = next(problem_pool)
                assert problem.n_problem() == 1
                iterval = self.library.infer_iteration_num(problem)[0]
                is_difficult = iterval > difficult_iter_threshold
                if is_difficult:
                    logger.debug("sampled! number: {}".format(len(difficult_problems)))
                    difficult_problems.append(problem)
                    pbar.update(1)
        return difficult_problems
