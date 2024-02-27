from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Type, TypeVar

import numpy as np

TaskT = TypeVar("TaskT", bound="TaskInterface")
PolicyT = TypeVar("PolicyT")


@dataclass
class TaskExpression:
    mat: Optional[np.ndarray]  # higher dimentional (2, 3, ...) data
    vec: Optional[np.ndarray]
    conditioned_vecs: List[np.ndarray]


class TaskInterface(ABC, Generic[TaskT, PolicyT]):
    @classmethod
    def predicated_sample(
        cls: Type[TaskT], n_inner: int, predicate: Callable[[TaskT], bool], max_trial: int
    ) -> Optional[TaskT]:
        for _ in range(max_trial):
            task = cls.sample(n_inner)
            if predicate(task):
                return task
        return None

    @classmethod
    @abstractmethod
    def sample(cls: Type[TaskT], n_inner: int) -> TaskT:
        ...

    @classmethod
    @abstractmethod
    def find_policy(cls: Type[TaskT], task: TaskT) -> PolicyT:
        # find the policy that solve the given task
        # in RL, return would be a policy network or DMP parameters
        # in planning, return would be a sequence of actions / paths
        ...

    @abstractmethod
    def export_task_expression(self) -> TaskExpression:
        # export that will be as datapoint of cost prediction network
        # we'l learn x->y mapping where x is task expression and y is cost
        ...

    @abstractmethod
    def to_parameters(self) -> np.ndarray:  # for serialization
        # convert the task to a 2D array (n_inner, param_dim)
        # use numpy for data efficiency.
        ...

    @classmethod
    @abstractmethod
    def from_parameters(cls: Type[TaskT], parameters: np.ndarray) -> TaskT:  # for deserialization
        ...


class RolloutInterface(ABC, Generic[TaskT, PolicyT]):
    @abstractmethod
    def admissible_max_cost(self) -> float:
        # if rollout cost exceeds this value, the rollout is considered failed
        ...

    @abstractmethod
    def rollout(self, policy: PolicyT, task: TaskT) -> List[float]:
        # do task with policy and return the costs
        # the reason why list is that task has multiple "inner" tasks
        ...
