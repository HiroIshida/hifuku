import logging
from dataclasses import dataclass
from typing import Callable, Generic, Iterator, Optional, Type, TypeVar, cast

import numpy as np
from rpbench.interface import TaskBase

logger = logging.getLogger(__name__)

ProblemT = TypeVar("ProblemT", bound=TaskBase)


@dataclass
class PredicatedProblemPool(Generic[ProblemT], Iterator[Optional[np.ndarray]]):
    problem_type: Type[ProblemT]
    n_problem_inner: int
    predicate: Callable[[ProblemT], bool]
    max_trial_factor: int

    def __next__(self) -> Optional[np.ndarray]:
        ret = self.problem_type.predicated_sample(
            self.n_problem_inner, self.predicate, self.max_trial_factor
        )
        if ret is None:
            return None
        return ret.to_intrinsic_desc_vecs()


@dataclass
class ProblemPool(Generic[ProblemT], Iterator[np.ndarray]):
    problem_type: Type[ProblemT]
    n_problem_inner: int

    def __next__(self) -> np.ndarray:
        return self.problem_type.sample(self.n_problem_inner).to_intrinsic_desc_vecs()

    def as_predicated(self) -> PredicatedProblemPool[ProblemT]:
        return cast(PredicatedProblemPool[ProblemT], self)

    def make_predicated(
        self, predicate: Callable[[ProblemT], bool], max_trial_factor: int
    ) -> PredicatedProblemPool[ProblemT]:
        return PredicatedProblemPool(
            self.problem_type, self.n_problem_inner, predicate, max_trial_factor
        )
