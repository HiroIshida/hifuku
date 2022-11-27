import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Sized, Type, cast

from hifuku.types import ProblemT

logger = logging.getLogger(__name__)


class ProblemPool(Iterable[ProblemT]):
    problem_type: Type[ProblemT]
    n_problem_inner: int


class PredicatedIteratorProblemPool(Iterator[Optional[ProblemT]], ProblemPool[ProblemT]):
    pass


class IteratorProblemPool(Iterator[ProblemT], ProblemPool[ProblemT], ABC):
    @abstractmethod
    def make_predicated(
        self, predicate: Callable[[ProblemT], bool], max_trial_factor: int
    ) -> PredicatedIteratorProblemPool[ProblemT]:
        pass

    def as_predicated(self) -> PredicatedIteratorProblemPool[ProblemT]:
        return cast(PredicatedIteratorProblemPool[ProblemT], self)


@dataclass
class SimplePredicatedProblemPool(PredicatedIteratorProblemPool[ProblemT]):
    problem_type: Type[ProblemT]
    predicate: Callable[[ProblemT], bool]
    max_trial_factor: int
    n_problem_inner: int = 1

    def __next__(self) -> Optional[ProblemT]:
        return self.problem_type.sample(
            self.n_problem_inner, predicate=self.predicate, max_trial_factor=self.max_trial_factor
        )


@dataclass
class SimpleProblemPool(IteratorProblemPool[ProblemT]):
    problem_type: Type[ProblemT]
    n_problem_inner: int = 1

    def __next__(self) -> ProblemT:
        return self.problem_type.sample(self.n_problem_inner)

    def make_predicated(
        self, predicate: Callable[[ProblemT], bool], max_trial_factor: int
    ) -> SimplePredicatedProblemPool[ProblemT]:
        return SimplePredicatedProblemPool(
            self.problem_type, predicate, max_trial_factor, self.n_problem_inner
        )


@dataclass
class TrivialIteratorPool(IteratorProblemPool[ProblemT]):
    iterator: Iterator[ProblemT]

    def __next__(self) -> ProblemT:
        return next(self.iterator)

    def make_predicated(
        self, predicate: Callable[[ProblemT], bool], max_trial_factor: int
    ) -> PredicatedIteratorProblemPool[ProblemT]:
        raise NotImplementedError("under construction")


class FixedProblemPool(Sized, ProblemPool[ProblemT]):
    @abstractmethod
    def __len__(self) -> int:
        ...

    def as_iterator(self) -> TrivialIteratorPool[ProblemT]:
        return TrivialIteratorPool(self.__iter__())


@dataclass
class SimpleFixedProblemPool(FixedProblemPool[ProblemT]):
    problem_type: Type[ProblemT]
    problem_list: List[ProblemT]
    n_problem_inner: int

    @classmethod
    def initialize(
        cls, problem_type: Type[ProblemT], n_problem: int, n_problem_inner: int = 1
    ) -> "SimpleFixedProblemPool[ProblemT]":
        problem_list = [problem_type.sample(n_problem_inner) for _ in range(n_problem)]
        return cls(problem_type, problem_list, n_problem_inner)

    def __len__(self) -> int:
        return len(self.problem_list)

    def __iter__(self) -> Iterator[ProblemT]:
        return self.problem_list.__iter__()
