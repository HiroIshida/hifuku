import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Iterator, Optional, Type

import numpy as np
import torch
from mohou.script_utils import create_default_logger
from mohou.trainer import TrainCache, TrainConfig, train
from skplan.solver.optimization import OsqpSqpPlanner

from hifuku.llazy.dataset import LazyDecomplessDataset
from hifuku.llazy.generation import DataGenerationTask, DataGenerationTaskArg
from hifuku.neuralnet import (
    IterationPredictor,
    IterationPredictorConfig,
    IterationPredictorDataset,
    VoxelAutoEncoder,
)
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.types import List, ProblemT, RawData, ResultProtocol


class ProblemPool(Iterator[ProblemT]):
    problem_type: Type[ProblemT]


@dataclass
class SimpleProblemPool(ProblemPool[ProblemT]):
    problem_type: Type[ProblemT]

    def __next__(self) -> ProblemT:
        return self.problem_type.sample(1)


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
        config = OsqpSqpPlanner.SolverConfig(verbose=False)
        problem = TabletopPlanningProblem.sample(n_pose=self.n_problem_inner)
        results = problem.solve(self.init_solution, config=config)
        print([r.nit for r in results])
        data = RawData.create(problem, results, self.init_solution, config)
        return data


class DatasetGenerator(Generic[ProblemT], ABC):
    problem_type: Type[ProblemT]
    cache_base_dir: Path

    def __init__(self, problem_type: Type[ProblemT], cache_base_dir: Optional[Path] = None):
        self.problem_type = problem_type
        if cache_base_dir is None:
            cache_base_dir = Path("/tmp/hifuku_dataset_cache")
        self.cache_base_dir = cache_base_dir

    @abstractmethod
    def generate(self, init_solution: np.ndarray, n_problem: int, n_problem_inner) -> Path:
        pass


class MultiProcessDatasetGenerator(DatasetGenerator[ProblemT]):
    n_process: int

    def __init__(
        self,
        problem_type: Type[ProblemT],
        cache_base_dir: Optional[Path] = None,
        n_process: Optional[int] = None,
    ):
        super().__init__(problem_type, cache_base_dir)
        if n_process is None:
            cpu_num = os.cpu_count()
            assert cpu_num is not None
            n_process = int(cpu_num * 0.5)
        self.n_process = n_process

    @staticmethod
    def split_number(num, div):
        return [num // div + (1 if x < num % div else 0) for x in range(div)]

    def generate(self, init_solution: np.ndarray, n_problem: int, n_problem_inner) -> Path:
        cache_dir_name = "{}-{}".format(self.problem_type.__name__, str(uuid.uuid4())[-8:])
        cache_dir_path = self.cache_base_dir / cache_dir_name
        cache_dir_path.mkdir(parents=True, exist_ok=False)

        n_problem_per_process_list = self.split_number(n_problem, self.n_process)

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
        return cache_dir_path


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

    def infer_iteration_num(self, problem: ProblemT) -> np.ndarray:
        # TODO: consider margin
        assert len(self.predictors) > 0

        mesh_np = problem.get_mesh()
        mesh = torch.from_numpy(np.expand_dims(mesh_np, axis=(0, 1))).float()

        desc_np = np.array(problem.get_descriptions())
        desc = torch.from_numpy(desc_np).float()
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


@dataclass
class SolutionLibraryGen(Generic[ProblemT], ABC):
    problem_type: Type[ProblemT]
    library: SolutionLibrary[ProblemT]
    difficult_threshold: float
    dataset_gen: Callable[[np.ndarray], Path]

    @classmethod
    def initialize(
        cls,
        problem_type: Type[ProblemT],
        ae_model: VoxelAutoEncoder,
        difficult_thrshold: float,
        dataset_gen: Callable[[np.ndarray], Path],
    ) -> "SolutionLibraryGen[ProblemT]":
        library = SolutionLibrary.initialize(problem_type, ae_model)
        return cls(problem_type, library, difficult_thrshold, dataset_gen)

    def learn_predictors(self, init_solution: np.ndarray, project_path: Path) -> IterationPredictor:
        pp = project_path
        dataset_path = self.dataset_gen(init_solution)
        dataset = IterationPredictorDataset.load(dataset_path, self.library.ae_model)
        raw_dataset = LazyDecomplessDataset.load(dataset_path, RawData, n_worker=-1)

        rawdata = raw_dataset.get_data(np.array([0]))[0]
        init_solution = rawdata.init_solution

        create_default_logger(pp, "iteration_predictor")
        model_conf = IterationPredictorConfig(12, self.library.ae_model.config.dim_bottleneck, 10)
        model = IterationPredictor(model_conf)
        model.initial_solution = init_solution
        tcache = TrainCache.from_model(model)
        train_conf = TrainConfig(n_epoch=100, batch_size=100)
        train(pp, tcache, dataset, train_conf)
        return model

    def _solve_problem(self, problem: ProblemT, n_trial: int) -> Optional[ResultProtocol]:
        assert problem.n_problem == 1
        for _ in range(n_trial):
            res = problem.solve()[0]
            if res.success:
                return res
        return None

    def step_active_sampling(self, project_path: Path, problem_pool: ProblemPool[ProblemT]):
        is_initialized = len(self.library.predictors) > 0
        if not is_initialized:
            problem = self.problem_type.create_standard()
            result = problem.solve()[0]
            assert result.success
            self.learn_predictors(result.x, project_path)
        else:
            difficult_problems: List[ProblemT] = []
            solution_candidates: List[np.ndarray] = []

            while len(difficult_problems) > 100:
                problem = next(problem_pool)
                assert problem.n_problem == 1
                iterval = self.library.infer_iteration_num(problem)[0]

                is_difficult = iterval > self.difficult_threshold
                if not is_difficult:
                    continue

                # try solve problem 5 trial
                res = self._solve_problem(problem, 5)
                if res is not None:  # seems feasible
                    difficult_problems.append(problem)
                    assert res.success
                    solution_candidates.append(res.x)

            score_list = []
            for solution_guess in solution_candidates:
                score = 0.0
                for problem in difficult_problems:
                    res = problem.solve(solution_guess)[0]
                    score += int(res.success)
                score_list.append(score)
            best_solution = solution_candidates[np.argmax(score_list)]
            self.learn_predictors(best_solution, project_path)
