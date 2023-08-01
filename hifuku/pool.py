"""
This module implements the multiprocess data generation that does not
fit into batch_solver and batch_sampler.

Because I don't have time, the interface is much different from batch_sampler
and batch_solvers. will be fixed someday.
"""

import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Iterator, List, Optional, Type, TypeVar, cast

import numpy as np

from hifuku.llazy.dataset import DatasetIterator, LazyDecomplessDataset
from hifuku.rpbench_wrap import PicklableSamplableBase, PicklableTaskBase

logger = logging.getLogger(__name__)

ProblemT = TypeVar("ProblemT", bound=PicklableTaskBase)
SamplableT = TypeVar("SamplableT", bound=PicklableSamplableBase)
OtherSamplableT = TypeVar("OtherSamplableT", bound=PicklableSamplableBase)
CachedProblemT = TypeVar("CachedProblemT", bound=PicklableTaskBase)  # TODO: rename to CacheTaskT
ProblemPoolT = TypeVar("ProblemPoolT", bound="ProblemPoolLike")
T = TypeVar("T")


class ProblemPoolLike(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def split(self: ProblemPoolT, n_split: int) -> List[ProblemPoolT]:
        ...

    @abstractmethod
    def parallelizable(self) -> bool:
        ...


class TypicalProblemPoolMixin:
    def reset(self) -> None:
        pass

    def split(self: ProblemPoolT, n_split: int) -> List[ProblemPoolT]:  # type: ignore[misc]
        return [copy.deepcopy(self) for _ in range(n_split)]

    def parallelizable(self) -> bool:
        return True


@dataclass
class PredicatedProblemPool(ProblemPoolLike, Iterator[Optional[SamplableT]]):
    problem_type: Type[SamplableT]
    n_problem_inner: int


@dataclass
class ProblemPool(ProblemPoolLike, Iterator[SamplableT]):
    problem_type: Type[SamplableT]
    n_problem_inner: int

    @abstractmethod
    def make_predicated(
        self, predicate: Callable[[SamplableT], bool], max_trial_factor: int
    ) -> PredicatedProblemPool[SamplableT]:
        pass

    def as_predicated(self) -> PredicatedProblemPool[SamplableT]:
        return cast(PredicatedProblemPool[SamplableT], self)


@dataclass
class TrivialPredicatedProblemPool(TypicalProblemPoolMixin, PredicatedProblemPool[SamplableT]):
    predicate: Callable[[SamplableT], bool]
    max_trial_factor: int

    def __next__(self) -> Optional[SamplableT]:
        return self.problem_type.predicated_sample(
            self.n_problem_inner, self.predicate, self.max_trial_factor
        )


@dataclass
class TrivialProblemPool(TypicalProblemPoolMixin, ProblemPool[SamplableT]):
    def __next__(self) -> SamplableT:
        return self.problem_type.sample(self.n_problem_inner)

    def make_predicated(
        self, predicate: Callable[[SamplableT], bool], max_trial_factor: int
    ) -> TrivialPredicatedProblemPool[SamplableT]:
        return TrivialPredicatedProblemPool(
            self.problem_type, self.n_problem_inner, predicate, max_trial_factor
        )


@dataclass
class PseudoIteratorPool(TypicalProblemPoolMixin, ProblemPool[SamplableT]):
    iterator: Iterator[SamplableT]

    def __next__(self) -> SamplableT:
        return next(self.iterator)

    def make_predicated(
        self, predicate: Callable[[SamplableT], bool], max_trial_factor: int
    ) -> PredicatedProblemPool[SamplableT]:
        raise NotImplementedError("under construction")

    def reset(self) -> None:
        pass

    def parallelizable(self) -> bool:
        return True


@dataclass
class CachedProblemPool(ProblemPool[SamplableT], Generic[OtherSamplableT, SamplableT]):
    """Use Cache of precomputed CacheProblemT, create pool
    of ProblemT.

    This pool is beneficial specifically when sampling the world (including cache)
    is not easy. In that case, if you have cache of other related tasks which can
    share the world, but not world-conditioned description, then you can use that
    cache as part of the task that you want to generate.
    """

    cache_path_list: List[Path]
    cache_problem_type: Type[OtherSamplableT]
    dataset_iter: Optional[DatasetIterator] = None

    def __post_init__(self):
        assert len(self.cache_path_list) > 0
        assert self.dataset_iter is None, "maybe reset is not called yet?"

    def reset(self):
        dataset = LazyDecomplessDataset[OtherSamplableT](
            self.cache_path_list, self.cache_problem_type, 1
        )
        self.dataset_iter = DatasetIterator(dataset)

    def split(self, n_split: int) -> List["CachedProblemPool[OtherSamplableT, SamplableT]"]:
        indices_list = np.array_split(np.arange(len(self.cache_path_list)), n_split)
        pools = []
        for indices in indices_list:
            paths = [self.cache_path_list[i] for i in indices]
            pool = CachedProblemPool(
                self.problem_type, self.n_problem_inner, paths, self.cache_problem_type, None
            )
            pools.append(pool)
        return pools

    def __next__(self) -> SamplableT:
        assert self.dataset_iter is not None
        data = next(self.dataset_iter)
        problem = self.problem_type.cast_from(data)
        if problem.n_inner_task > 0:
            assert problem.n_inner_task == self.n_problem_inner
            return problem
        elif problem.n_inner_task == 0:
            descs = problem.sample_descriptions(problem.world, self.n_problem_inner)
            assert descs is not None  # TODO: due to change in rpbench, we must handle this case
            return self.problem_type(problem.world, descs, problem._cache)
        else:
            assert False

    @classmethod
    def load(
        cls,
        cached_problem_type: Type[OtherSamplableT],
        problem_type: Type[SamplableT],
        n_problem_inner: int,
        cache_dir_path: Path,
    ):
        cache_path_list = [p for p in cache_dir_path.iterdir() if p.name.endswith(".gz")]
        return cls(problem_type, n_problem_inner, cache_path_list, cached_problem_type)

    def make_predicated(
        self, predicate: Callable[[SamplableT], bool], max_trial_factor: int
    ) -> PredicatedProblemPool[SamplableT]:
        raise NotImplementedError("under construction")

    def parallelizable(self) -> bool:
        return False
