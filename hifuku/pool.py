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
from typing import Callable, Iterator, List, Optional, Type, TypeVar, cast

from rpbench.interface import SamplableBase, TaskBase

logger = logging.getLogger(__name__)

ProblemT = TypeVar("ProblemT", bound=TaskBase)
SamplableT = TypeVar("SamplableT", bound=SamplableBase)
OtherSamplableT = TypeVar("OtherSamplableT", bound=TaskBase)
CachedProblemT = TypeVar("CachedProblemT", bound=SamplableBase)  # TODO: rename to CacheTaskT
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
