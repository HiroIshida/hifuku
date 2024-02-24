"""
This module implements the multiprocess data generation that does not
fit into batch_solver and batch_sampler.

Because I don't have time, the interface is much different from batch_sampler
and batch_solvers. will be fixed someday.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Type, TypeVar, Union, cast

from rpbench.interface import TaskBase

logger = logging.getLogger(__name__)

ProblemT = TypeVar("ProblemT", bound=TaskBase)
ProblemPoolT = TypeVar("ProblemPoolT", bound=Union["PredicatedProblemPool", "ProblemPool"])


@dataclass
class PredicatedProblemPool(Iterator[Optional[ProblemT]]):
    problem_type: Type[ProblemT]
    n_problem_inner: int
    predicate: Callable[[ProblemT], bool]
    max_trial_factor: int

    def __next__(self) -> Optional[ProblemT]:
        return self.problem_type.predicated_sample(
            self.n_problem_inner, self.predicate, self.max_trial_factor
        )


@dataclass
class ProblemPool(Iterator[ProblemT]):
    problem_type: Type[ProblemT]
    n_problem_inner: int

    def __next__(self) -> ProblemT:
        return self.problem_type.sample(self.n_problem_inner)

    def as_predicated(self) -> PredicatedProblemPool[ProblemT]:
        return cast(PredicatedProblemPool[ProblemT], self)

    def make_predicated(
        self, predicate: Callable[[ProblemT], bool], max_trial_factor: int
    ) -> PredicatedProblemPool[ProblemT]:
        return PredicatedProblemPool(
            self.problem_type, self.n_problem_inner, predicate, max_trial_factor
        )
