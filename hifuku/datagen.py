import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, Type

import numpy as np

from hifuku.llazy.generation import DataGenerationTask, DataGenerationTaskArg
from hifuku.threedim.tabletop import TabletopPlanningProblem
from hifuku.types import ProblemT, RawData

logger = logging.getLogger(__name__)


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
        logger.debug("generated single data")
        logger.debug("success: {}".format([r.success for r in results]))
        logger.debug("iteration: {}".format([r.nit for r in results]))
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
            logger.info("n_process is not set. automatically determine")
            cpu_num = os.cpu_count()
            assert cpu_num is not None
            n_process = int(cpu_num * 0.5)
        logger.info("n_process is set to {}".format(n_process))
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
