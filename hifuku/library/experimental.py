import copy
import logging
import multiprocessing
import os
import pickle
import random
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, List, Optional

import numpy as np
import pyclustering.cluster.xmeans as xmeans
import torch
import tqdm
from skmp.solver.interface import ConfigT, ResultT
from skmp.trajectory import Trajectory

from hifuku.classifier import SVM, SVMDataset
from hifuku.datagen import split_number
from hifuku.library.core import SolutionLibrary, _SolutionLibrarySampler
from hifuku.pool import ProblemPool, ProblemT, PseudoIteratorPool
from hifuku.utils import num_torch_thread

logger = logging.getLogger(__name__)


@dataclass
class LargestDifficultClusterPredicate(Generic[ProblemT, ConfigT, ResultT]):
    library: SolutionLibrary[ProblemT, ConfigT, ResultT]
    svm: SVM
    accept_proba_threshold: float

    def __post_init__(self):
        assert self.library.device == torch.device("cpu")

    @classmethod
    def create(
        cls,
        library: SolutionLibrary[ProblemT, ConfigT, ResultT],
        difficult_problems: List[ProblemT],
        ambient_problems: List[ProblemT],
        accept_proba_threshold: float = 0.4,
    ) -> "LargestDifficultClusterPredicate[ProblemT, ConfigT, ResultT]":
        """
        difficult problems for detect the largest cluster
        ambient_problems + difficult_problems for fit the clf
        """

        # sanity check (only first element)c:
        assert difficult_problems[0].n_inner_task == 1
        assert ambient_problems[0].n_inner_task == 1

        # lirary should be put on cpu
        cpu_device = torch.device("cpu")
        if library.device != cpu_device:
            logger.debug("library is on gpu. copy and put the library on cpu")
            library = copy.deepcopy(library)
            library.put_on_device(cpu_device)

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
        assert problem.n_inner_task == 1
        iters = self.library._infer_iteration_num(problem).flatten()
        proba = self.svm.predict_proba(iters)
        return proba > self.accept_proba_threshold

    @staticmethod
    def task(
        library: SolutionLibrary[ProblemT, ConfigT, ResultT],
        svm: SVM,
        n_sample: int,
        pool: ProblemPool[ProblemT],
        accept_threshold: float,
        ambient_rate: float,
        show_progress_bar: bool,
        n_thread: int,
        cache_path: Path,
    ) -> None:
        def predicate(problem: ProblemT) -> bool:
            assert problem.n_inner_task == 1
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
        pool: ProblemPool[ProblemT],
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


class ClusterBasedSolutionLibrarySampler(_SolutionLibrarySampler[ProblemT, ConfigT, ResultT]):
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

    def _determine_init_solution(self) -> Trajectory:
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
            iter_pool = PseudoIteratorPool(self.problem_type, 1, problems_in_clf.__iter__())
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
        assert best_solution is not None, "TODO: repeat if no best solution found"
        return best_solution
