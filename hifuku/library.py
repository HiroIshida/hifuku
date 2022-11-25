import copy
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

from hifuku.datagen import BatchProblemSolver
from hifuku.llazy.dataset import LazyDecomplessDataset
from hifuku.margin import CoverageResult
from hifuku.neuralnet import (
    IterationPredictor,
    IterationPredictorConfig,
    IterationPredictorDataset,
    VoxelAutoEncoder,
)
from hifuku.types import ProblemInterface, ProblemT, RawData, ResultProtocol

logger = logging.getLogger(__name__)


class MultiProcessProblemSolver:
    """
    This is similar to MultiProcessDatasetGenerator but whilt MultiProcessDatasetGenerator
    samples problems inside, this solver will solve problem specified.
    """

    @dataclass
    class ProblemSolverArg:
        indices: np.ndarray
        problems: Sequence[ProblemInterface]
        init_solutions: Sequence[np.ndarray]
        disable_tqdm: bool

    @staticmethod
    def _solve(arg: ProblemSolverArg, q: multiprocessing.Queue):
        with tqdm.tqdm(total=len(arg.problems), disable=arg.disable_tqdm) as pbar:
            for idx, problem, init_solution in zip(arg.indices, arg.problems, arg.init_solutions):
                assert problem.n_problem() == 1
                result = problem.solve(init_solution)[0]
                q.put((idx, result))
                pbar.update(1)

    @classmethod
    def solve(
        cls,
        problems: Sequence[ProblemInterface],
        init_solutions: Sequence[np.ndarray],
        n_process: Optional[int],
    ) -> Sequence[ResultProtocol]:

        assert len(problems) == len(init_solutions)

        if n_process is None:
            cpu_count = os.cpu_count()
            assert cpu_count is not None
            n_process = int(0.5 * cpu_count)

        n_process = min(n_process, len(problems))
        logger.debug("*n_process: {}".format(n_process))

        is_single_process = n_process == 1
        if is_single_process:
            results = []
            maxiter = problems[0].get_solver_config().maxiter
            logger.debug("*maxiter: {}".format(maxiter))
            for problem, init_solution in zip(problems, init_solutions):
                result = problem.solve(init_solution)[0]
                results.append(result)
            return results
        else:
            indices = np.array(list(range(len(problems))))
            indices_list_per_worker = np.array_split(indices, n_process)

            q = multiprocessing.Queue()  # type: ignore
            indices_list_per_worker = np.array_split(indices, n_process)

            process_list = []
            for i, indices_part in enumerate(indices_list_per_worker):
                disable_tqdm = i > 0
                problems_part = [problems[idx] for idx in indices_part]
                init_solutions_part = [init_solutions[idx] for idx in indices_part]
                arg = cls.ProblemSolverArg(
                    indices_part, problems_part, init_solutions_part, disable_tqdm
                )
                p = multiprocessing.Process(target=cls._solve, args=(arg, q))
                p.start()
                process_list.append(p)

            idx_result_pairs = [q.get() for _ in range(len(problems))]
            idx_result_pairs_sorted = sorted(idx_result_pairs, key=lambda x: x[0])  # type: ignore
            _, results = zip(*idx_result_pairs_sorted)
            return list(results)


class ProblemPool(Iterable[ProblemT]):
    problem_type: Type[ProblemT]


class IteratorProblemPool(Iterator[ProblemT], ProblemPool[ProblemT]):
    pass


@dataclass
class SimpleProblemPool(IteratorProblemPool[ProblemT]):
    problem_type: Type[ProblemT]
    n_problem_inner: int = 1

    def __next__(self) -> ProblemT:
        return self.problem_type.sample(self.n_problem_inner)


class FixedProblemPool(Sized, ProblemPool[ProblemT]):
    @abstractmethod
    def __len__(self) -> int:
        ...


@dataclass
class SimpleFixedProblemPool(FixedProblemPool[ProblemT]):
    problem_type: Type[ProblemT]
    problem_list: List[ProblemT]

    @classmethod
    def initialize(
        cls, problem_type: Type[ProblemT], n_problem: int
    ) -> "SimpleFixedProblemPool[ProblemT]":
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

    def measure_full_coverage(self, problem_pool: FixedProblemPool[ProblemT]) -> CoverageResult:
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
        results = MultiProcessProblemSolver.solve(problems, init_solution_est_list, None)

        maxiter = self.problem_type.get_solver_config().maxiter
        iterval_real_list = [(maxiter if not r.success else r.nit) for r in results]

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
    solver: BatchProblemSolver
    config: LibrarySamplerConfig
    validation_problem_pool: FixedProblemPool[ProblemT]

    @classmethod
    def initialize(
        cls,
        problem_type: Type[ProblemT],
        ae_model: VoxelAutoEncoder,
        solver: BatchProblemSolver,
        config: LibrarySamplerConfig,
        validation_problem_pool: FixedProblemPool[ProblemT],
    ) -> "SimpleSolutionLibrarySampler[ProblemT]":
        library = SolutionLibrary.initialize(
            problem_type, ae_model, config.solvable_threshold_factor
        )
        logger.info("library sampler config: {}".format(config))
        return cls(problem_type, library, solver, config, validation_problem_pool)

    @abstractmethod
    def step_active_sampling(
        self,
        project_path: Path,
        problem_pool_dataset: Optional[IteratorProblemPool[ProblemT]] = None,
        problem_pool_init_traj: Optional[IteratorProblemPool[ProblemT]] = None,
    ) -> None:
        ...

    @abstractmethod
    def _determine_init_solution(self, problem_pool: IteratorProblemPool[ProblemT]) -> np.ndarray:
        ...

    def learn_predictors(
        self,
        init_solution: np.ndarray,
        project_path: Path,
        problem_pool: IteratorProblemPool[ProblemT],
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
        problems = [next(problem_pool) for _ in range(self.config.n_problem)]
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
        difficult_iter_threshold: float,
    ) -> List[np.ndarray]:

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
        difficult_iter_threshold: float,
    ) -> List[ProblemT]:

        difficult_problems: List[ProblemT] = []
        with tqdm.tqdm(total=n_sample) as pbar:
            while len(difficult_problems) < n_sample:
                logger.debug("try sampling difficutl problem...")
                problem = next(problem_pool)
                assert problem.n_problem() == 1
                infer_res = self.library.infer(problem)[0]
                iterval = infer_res.nit
                is_difficult = iterval > difficult_iter_threshold
                if is_difficult:
                    logger.debug("sampled! number: {}".format(len(difficult_problems)))
                    difficult_problems.append(problem)
                    pbar.update(1)
        return difficult_problems


class SimpleSolutionLibrarySampler(_SolutionLibrarySampler[ProblemT]):
    def step_active_sampling(
        self,
        project_path: Path,
        problem_pool_dataset: Optional[IteratorProblemPool[ProblemT]] = None,
        problem_pool_init_traj: Optional[IteratorProblemPool[ProblemT]] = None,
    ) -> None:
        logger.info("active sampling step")

        if problem_pool_dataset is None:
            logger.info("problem pool is not specified. use SimpleProblemPool")
            # TODO: smelling! n_problem_inner should not be set here
            problem_pool_dataset = SimpleProblemPool(self.problem_type, self.config.n_problem_inner)

        if problem_pool_init_traj is None:
            logger.info("problem pool is not specified. use SimpleProblemPool")
            problem_pool_init_traj = SimpleProblemPool(self.problem_type, 1)

        init_solution = self._determine_init_solution(problem_pool_init_traj)
        predictor = self.learn_predictors(init_solution, project_path, problem_pool_dataset)

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
        coverage_result = singleton_library.measure_full_coverage(self.validation_problem_pool)
        logger.info(coverage_result)
        margin = coverage_result.determine_margin(self.config.acceptable_false_positive_rate)

        logger.info("margin is set to {}".format(margin))
        self.library.add(predictor, margin, coverage_result)

        coverage = self.library.measure_coverage(self.validation_problem_pool)
        logger.info("current library's coverage estimate: {}".format(coverage))

        self.library.dump(project_path)

    def _determine_init_solution(self, problem_pool: IteratorProblemPool[ProblemT]) -> np.ndarray:

        is_initialized = len(self.library.predictors) > 0
        if not is_initialized:
            logger.info("start determine init solution using standard problem")
            init_solution = self.problem_type.get_default_init_solution()
            return init_solution
        else:
            logger.info("start determine init solution len(lib) > 0")

            logger.info("sample solution candidates")
            difficult_iter_threshold = (
                self.problem_type.get_solver_config().maxiter
                * self.config.difficult_threshold_factor
            )
            solution_candidates = self._sample_solution_canidates(
                self.config.n_difficult_problem, problem_pool, difficult_iter_threshold
            )

            logger.info("sample difficult problems")
            difficult_problems = self._sample_difficult_problems(
                self.config.n_difficult_problem, problem_pool, difficult_iter_threshold
            )

            logger.info("compute scores")
            score_list = []
            maxiter = self.problem_type.get_solver_config().maxiter
            for solution_guess in solution_candidates:
                solution_guesses = [solution_guess] * len(difficult_problems)
                results = MultiProcessProblemSolver.solve(
                    difficult_problems, solution_guesses, None
                )
                iterval_real_list = [(maxiter if not r.success else r.nit) for r in results]
                score = -sum(iterval_real_list)  # must be nagative
                logger.debug("*score of solution cand: {}".format(score))
                score_list.append(score)

            best_idx = np.argmax(score_list)
            best_solution = solution_candidates[best_idx]
            logger.debug("best score: {}".format(score_list[best_idx]))
            return best_solution
