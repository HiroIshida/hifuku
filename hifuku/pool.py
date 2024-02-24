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

from rpbench.interface import TaskBase

logger = logging.getLogger(__name__)

ProblemT = TypeVar("ProblemT", bound=TaskBase)
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
class PredicatedProblemPool(ProblemPoolLike, Iterator[Optional[ProblemT]]):
    problem_type: Type[ProblemT]
    n_problem_inner: int


@dataclass
class ProblemPool(ProblemPoolLike, Iterator[ProblemT]):
    problem_type: Type[ProblemT]
    n_problem_inner: int

    @abstractmethod
    def make_predicated(
        self, predicate: Callable[[ProblemT], bool], max_trial_factor: int
    ) -> PredicatedProblemPool[ProblemT]:
        pass

    def as_predicated(self) -> PredicatedProblemPool[ProblemT]:
        return cast(PredicatedProblemPool[ProblemT], self)


@dataclass
class TrivialPredicatedProblemPool(TypicalProblemPoolMixin, PredicatedProblemPool[ProblemT]):
    predicate: Callable[[ProblemT], bool]
    max_trial_factor: int

    def __next__(self) -> Optional[ProblemT]:
        return self.problem_type.predicated_sample(
            self.n_problem_inner, self.predicate, self.max_trial_factor
        )


@dataclass
class TrivialProblemPool(TypicalProblemPoolMixin, ProblemPool[ProblemT]):
    def __next__(self) -> ProblemT:
        return self.problem_type.sample(self.n_problem_inner)

    def make_predicated(
        self, predicate: Callable[[ProblemT], bool], max_trial_factor: int
    ) -> TrivialPredicatedProblemPool[ProblemT]:
        return TrivialPredicatedProblemPool(
            self.problem_type, self.n_problem_inner, predicate, max_trial_factor
        )


@dataclass
class PseudoIteratorPool(TypicalProblemPoolMixin, ProblemPool[ProblemT]):
    iterator: Iterator[ProblemT]

    def __next__(self) -> ProblemT:
        return next(self.iterator)

    def make_predicated(
        self, predicate: Callable[[ProblemT], bool], max_trial_factor: int
    ) -> PredicatedProblemPool[ProblemT]:
        raise NotImplementedError("under construction")

    def reset(self) -> None:
        pass

    def parallelizable(self) -> bool:
        return True
