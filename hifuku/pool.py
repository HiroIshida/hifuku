import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Type, cast

from hifuku.types import ProblemT

logger = logging.getLogger(__name__)


@dataclass
class PredicatedProblemPool(ABC, Iterator[Optional[ProblemT]]):
    problem_type: Type[ProblemT]
    n_problem_inner: int

    @abstractmethod
    def reset(self) -> None:
        ...


@dataclass
class ProblemPool(Iterator[ProblemT], ABC):
    problem_type: Type[ProblemT]
    n_problem_inner: int

    @abstractmethod
    def make_predicated(
        self, predicate: Callable[[ProblemT], bool], max_trial_factor: int
    ) -> PredicatedProblemPool[ProblemT]:
        pass

    def as_predicated(self) -> PredicatedProblemPool[ProblemT]:
        return cast(PredicatedProblemPool[ProblemT], self)

    @abstractmethod
    def reset(self) -> None:
        ...


@dataclass
class TrivialPredicatedProblemPool(PredicatedProblemPool[ProblemT]):
    predicate: Callable[[ProblemT], bool]
    max_trial_factor: int

    def __next__(self) -> Optional[ProblemT]:
        return self.problem_type.sample(
            self.n_problem_inner, predicate=self.predicate, max_trial_factor=self.max_trial_factor
        )

    def reset(self) -> None:
        pass


@dataclass
class TrivialProblemPool(ProblemPool[ProblemT]):
    def __next__(self) -> ProblemT:
        return self.problem_type.sample(self.n_problem_inner)

    def make_predicated(
        self, predicate: Callable[[ProblemT], bool], max_trial_factor: int
    ) -> TrivialPredicatedProblemPool[ProblemT]:
        return TrivialPredicatedProblemPool(
            self.problem_type, self.n_problem_inner, predicate, max_trial_factor
        )

    def reset(self) -> None:
        pass


@dataclass
class PseudoIteratorPool(ProblemPool[ProblemT]):
    iterator: Iterator[ProblemT]

    def __next__(self) -> ProblemT:
        return next(self.iterator)

    def make_predicated(
        self, predicate: Callable[[ProblemT], bool], max_trial_factor: int
    ) -> PredicatedProblemPool[ProblemT]:
        raise NotImplementedError("under construction")

    def reset(self) -> None:
        pass
