import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterable, Iterator, Optional, Sized, Type

import numpy as np
import torch
import tqdm
from mohou.script_utils import create_default_logger
from mohou.trainer import TrainCache, TrainConfig, train

from hifuku.guarantee.margin import CoverageResult
from hifuku.llazy.dataset import LazyDecomplessDataset
from hifuku.llazy.generation import DataGenerationTask, DataGenerationTaskArg
from hifuku.neuralnet import (
    IterationPredictor,
    IterationPredictorConfig,
    IterationPredictorDataset,
    VoxelAutoEncoder,
)
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.types import List, ProblemT, RawData


class ProblemPool(Iterable[ProblemT]):
    problem_type: Type[ProblemT]


@dataclass
class SimpleProblemPool(Iterator[ProblemT], ProblemPool[ProblemT]):
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


class HifukuDataGenerationTask(DataGenerationTask[RawData]):
    n_problem_inner: int
    init_solution: np.ndarray

    def __init__(self, arg: DataGenerationTaskArg, n_problem_inner: int, init_solution: np.ndarray):
        super().__init__(arg)
        self.n_problem_inner = n_problem_inner
        self.init_solution = init_solution

    def post_init_hook(self) -> None:
        pass

    def generate_single_data(self) -> RawData:
        problem = TabletopPlanningProblem.sample(n_pose=self.n_problem_inner)
        results = problem.solve(self.init_solution)
        print([r.nit for r in results])
        data = RawData.create(problem, results, self.init_solution)
        return data


class DatasetGenerator(Generic[ProblemT], ABC):
    problem_type: Type[ProblemT]

    def __init__(self, problem_type: Type[ProblemT], cache_base_dir: Optional[Path] = None):
        self.problem_type = problem_type

    @abstractmethod
    def generate(
        self, init_solution: np.ndarray, n_problem: int, n_problem_inner, cache_dir_path: Path
    ) -> None:
        pass


class MultiProcessDatasetGenerator(DatasetGenerator[ProblemT]):
    n_process: int

    def __init__(self, problem_type: Type[ProblemT], n_process: Optional[int] = None):
        super().__init__(problem_type)
        if n_process is None:
            cpu_num = os.cpu_count()
            assert cpu_num is not None
            n_process = int(cpu_num * 0.5)
        self.n_process = n_process

    @staticmethod
    def split_number(num, div):
        return [num // div + (1 if x < num % div else 0) for x in range(div)]

    def generate(
        self, init_solution: np.ndarray, n_problem: int, n_problem_inner, cache_dir_path: Path
    ) -> None:
        n_problem_per_process_list = self.split_number(n_problem, self.n_process)
        assert cache_dir_path.exists()

        if self.n_process > 1:
            process_list = []
            for idx_process, n_problem_per_process in enumerate(n_problem_per_process_list):
                show_process_bar = idx_process == 1
                arg = DataGenerationTaskArg(
                    n_problem_per_process, show_process_bar, cache_dir_path, extension=".npz"
                )
                p = HifukuDataGenerationTask(arg, n_problem_inner, init_solution)
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()
        else:
            arg = DataGenerationTaskArg(n_problem, True, cache_dir_path, extension=".npz")
            task = HifukuDataGenerationTask(arg, n_problem_inner, init_solution)
            task.run()


@dataclass
class SolutionLibrary(Generic[ProblemT]):
    problem_type: Type[ProblemT]
    ae_model: VoxelAutoEncoder
    predictors: List[IterationPredictor]
    margins: List[float]

    @classmethod
    def initialize(
        cls, problem_type: Type[ProblemT], ae_model: VoxelAutoEncoder
    ) -> "SolutionLibrary[ProblemT]":
        return cls(problem_type, ae_model, [], [])

    @property
    def device(self) -> torch.device:
        return self.ae_model.device

    def infer_iteration_num(self, problem: ProblemT) -> np.ndarray:
        # TODO: consider margin
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
            itervals_np = itervals.detach().cpu().numpy() + margin
            itervals_list.append(itervals_np)
        itervals_min = np.min(np.array(itervals_list), axis=0)
        return itervals_min

    def add(self, predictor: IterationPredictor, margin: float):
        self.predictors.append(predictor)
        self.margins.append(margin)


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

    @classmethod
    def initialize(
        cls,
        problem_type: Type[ProblemT],
        ae_model: VoxelAutoEncoder,
        dataset_gen: DatasetGenerator,
        config: LibrarySamplerConfig,
        validation_problem_pool: FixedProblemPool[ProblemT],
    ) -> "SolutionLibrarySampler[ProblemT]":
        library = SolutionLibrary.initialize(problem_type, ae_model)
        return cls(problem_type, library, dataset_gen, config, validation_problem_pool)

    def step_active_sampling(
        self, project_path: Path, problem_pool: Optional[ProblemPool[ProblemT]] = None
    ):

        if problem_pool is None:
            problem_pool = SimpleProblemPool(self.problem_type)

        init_solution = self._determine_init_solution(problem_pool)
        predictor = self.learn_predictors(init_solution, project_path)

        singleton_library = SolutionLibrary(
            self.problem_type, self.library.ae_model, [predictor], [0.0]
        )

        iterval_est_list = []
        iterval_real_list = []
        maxiter = self.problem_type.get_solver_config().maxiter
        for problem in self.validation_problem_pool:
            assert problem.n_problem() == 1
            iterval_est = singleton_library.infer_iteration_num(problem)[0].item()
            iterval_est_list.append(iterval_est)

            # hmm, it's bit dirty that the following clamping is applied here and also in RawData
            result = problem.solve(init_solution)[0]
            iterval_real = result.nit if result.success else maxiter
            iterval_real_list.append(iterval_real)

        success_iter_threshold = maxiter * self.config.difficult_threshold_factor
        coverage_result = CoverageResult(
            np.array(iterval_real_list), np.array(iterval_est_list), success_iter_threshold
        )
        margin = coverage_result.determine_margin(self.config.acceptable_false_positive_rate)
        self.library.add(predictor, margin)

    def learn_predictors(self, init_solution: np.ndarray, project_path: Path) -> IterationPredictor:
        pp = project_path
        cache_dir_base = pp / "dataset_cache"
        cache_dir_base.mkdir(exist_ok=True, parents=True)

        cache_dir_name = "{}-{}".format(self.problem_type.__name__, str(uuid.uuid4())[-8:])
        cache_dir_path = cache_dir_base / cache_dir_name
        assert not cache_dir_path.exists()
        cache_dir_path.mkdir()

        self.dataset_gen.generate(
            init_solution, self.config.n_problem, self.config.n_problem_inner, cache_dir_path
        )
        dataset = IterationPredictorDataset.load(cache_dir_path, self.library.ae_model)
        raw_dataset = LazyDecomplessDataset.load(cache_dir_path, RawData, n_worker=-1)

        rawdata = raw_dataset.get_data(np.array([0]))[0]
        init_solution = rawdata.init_solution

        create_default_logger(pp, "iteration_predictor")
        model_conf = IterationPredictorConfig(12, self.library.ae_model.config.dim_bottleneck, 10)
        model = IterationPredictor(model_conf)
        model.initial_solution = init_solution
        tcache = TrainCache.from_model(model)
        train(pp, tcache, dataset, self.config.train_config)
        return model

    def _determine_init_solution(self, problem_pool: ProblemPool[ProblemT]) -> np.ndarray:

        is_initialized = len(self.library.predictors) > 0
        if not is_initialized:
            for _ in range(20):
                try:
                    problem = self.problem_type.create_standard()
                    result = problem.solve()[0]
                    assert result.success
                    return result.x
                except self.problem_type.SamplingBasedInitialguessFail:
                    pass
            # assumes that standared problem is easy enough and must be solved
            assert False
        else:
            solution_candidates = self._sample_solution_canidates(problem_pool)
            difficult_problems = self._sample_difficult_problems(problem_pool)

            # then, sample difficult problems
            score_list = []
            for solution_guess in solution_candidates:
                score = 0.0
                for problem in difficult_problems:
                    res = problem.solve(solution_guess)[0]
                    score += int(res.success)
                score_list.append(score)
            best_solution = solution_candidates[np.argmax(score_list)]
            return best_solution

    def _sample_solution_canidates(self, problem_pool: ProblemPool[ProblemT]) -> List[np.ndarray]:
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
                    print("try solving...")
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
                        pbar.update(1)
        return solution_candidates

    def _sample_difficult_problems(self, problem_pool: ProblemPool[ProblemT]) -> List[ProblemT]:
        maxiter = self.problem_type.get_solver_config().maxiter
        difficult_iter_threshold = maxiter * self.config.difficult_threshold_factor

        difficult_problems: List[ProblemT] = []
        with tqdm.tqdm(total=self.config.n_difficult_problem) as pbar:
            while len(difficult_problems) < self.config.n_difficult_problem:
                problem = next(problem_pool)
                assert problem.n_problem() == 1
                iterval = self.library.infer_iteration_num(problem)[0]
                is_difficult = iterval > difficult_iter_threshold
                if is_difficult:
                    difficult_problems.append(problem)
                    pbar.update(1)
        return difficult_problems
