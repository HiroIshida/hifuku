import logging
from dataclasses import dataclass
from typing import Callable, Generic, Iterator, Optional, Type, TypeVar, cast

import numpy as np
from rpbench.interface import TaskBase

logger = logging.getLogger(__name__)

TaskT = TypeVar("TaskT", bound=TaskBase)


@dataclass
class PredicatedTaskPool(Generic[TaskT], Iterator[Optional[np.ndarray]]):
    task_type: Type[TaskT]
    n_task_inner: int
    predicate: Callable[[TaskT], bool]
    max_trial_factor: int

    def __next__(self) -> Optional[np.ndarray]:
        ret = self.task_type.predicated_sample(
            self.n_task_inner, self.predicate, self.max_trial_factor
        )
        if ret is None:
            return None
        return ret.to_task_params()


@dataclass
class TaskPool(Generic[TaskT], Iterator[np.ndarray]):
    task_type: Type[TaskT]
    n_task_inner: int

    def __next__(self) -> np.ndarray:
        return self.task_type.sample(self.n_task_inner).to_task_params()

    def as_predicated(self) -> PredicatedTaskPool[TaskT]:
        return cast(PredicatedTaskPool[TaskT], self)

    def make_predicated(
        self, predicate: Callable[[TaskT], bool], max_trial_factor: int
    ) -> PredicatedTaskPool[TaskT]:
        return PredicatedTaskPool(self.task_type, self.n_task_inner, predicate, max_trial_factor)
