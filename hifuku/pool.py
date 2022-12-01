import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Type, cast

from hifuku.types import ProblemT

logger = logging.getLogger(__name__)


@dataclass
class PredicatedIteratorProblemPool(Iterator[Optional[ProblemT]]):
    problem_type: Type[ProblemT]
    n_problem_inner: int


@dataclass
class IteratorProblemPool(Iterator[ProblemT], ABC):
    problem_type: Type[ProblemT]
    n_problem_inner: int

    @abstractmethod
    def make_predicated(
        self, predicate: Callable[[ProblemT], bool], max_trial_factor: int
    ) -> PredicatedIteratorProblemPool[ProblemT]:
        pass

    def as_predicated(self) -> PredicatedIteratorProblemPool[ProblemT]:
        return cast(PredicatedIteratorProblemPool[ProblemT], self)


@dataclass
class SimplePredicatedIteratorProblemPool(PredicatedIteratorProblemPool[ProblemT]):
    predicate: Callable[[ProblemT], bool]
    max_trial_factor: int

    def __next__(self) -> Optional[ProblemT]:
        return self.problem_type.sample(
            self.n_problem_inner, predicate=self.predicate, max_trial_factor=self.max_trial_factor
        )


@dataclass
class SimpleIteratorProblemPool(IteratorProblemPool[ProblemT]):
    def __next__(self) -> ProblemT:
        return self.problem_type.sample(self.n_problem_inner)

    def make_predicated(
        self, predicate: Callable[[ProblemT], bool], max_trial_factor: int
    ) -> SimplePredicatedIteratorProblemPool[ProblemT]:
        return SimplePredicatedIteratorProblemPool(
            self.problem_type, self.n_problem_inner, predicate, max_trial_factor
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
