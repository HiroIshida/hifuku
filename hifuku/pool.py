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
    predicate: Callable[[TaskT], bool]
    timeout: int = 60

    def __next__(self) -> Optional[np.ndarray]:
        try:
            ret = self.task_type.sample(predicate=self.predicate, timeout=self.timeout)
            return ret.to_task_param()
        except TimeoutError:
            # TODO: is it better to return None or raise an exception?
            logger.warning(f"TimeoutError in {self.task_type.sample.__name__}")
            return None

        if ret is None:
            return None
        return ret.to_task_params()


@dataclass
class PredicatedTaskPool2(Generic[TaskT], Iterator[Optional[np.ndarray]]):
    # 2024/9/15: this is adhoc modification.
    # to measure to measure total evaluation of predicate()
    # in sampler
    task_type: Type[TaskT]
    predicate: Callable[[TaskT], bool]
    timeout: int = 60

    def __next__(self) -> Optional[np.ndarray]:
        try:
            ret = self.task_type.sample(predicate=self.predicate, timeout=self.timeout)
            if self.predicate(ret):
                return ret.to_task_param()
            else:
                return None
        except TimeoutError:
            # TODO: is it better to return None or raise an exception?
            logger.warning(f"TimeoutError in {self.task_type.sample.__name__}")
            return None

        if ret is None:
            return None
        return ret.to_task_params()


@dataclass
class TaskPool(Generic[TaskT], Iterator[np.ndarray]):
    task_type: Type[TaskT]

    def __next__(self) -> np.ndarray:
        return self.task_type.sample().to_task_param()

    def as_predicated(self) -> PredicatedTaskPool[TaskT]:
        return cast(PredicatedTaskPool[TaskT], self)

    def make_predicated(
        self,
        predicate: Callable[[TaskT], bool],
        timeout: int,
    ) -> PredicatedTaskPool[TaskT]:
        return PredicatedTaskPool(self.task_type, predicate, timeout)

    def make_predicated2(
        self,
        predicate: Callable[[TaskT], bool],
    ) -> PredicatedTaskPool2[TaskT]:
        return PredicatedTaskPool2(self.task_type, predicate)
