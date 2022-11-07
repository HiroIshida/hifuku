from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, List, Optional, Protocol, Type, TypeVar

import numpy as np

ProblemT = TypeVar("ProblemT", bound="ProblemInterface")
ResultT = TypeVar("ResultT", bound="ResultProtocol")


class ResultProtocol(Protocol):
    nit: int
    success: bool


class ProblemInterface(ABC):
    @classmethod
    @abstractmethod
    def sample(cls: Type[ProblemT]) -> ProblemT:
        ...

    @abstractmethod
    def solve(self, sol_init: np.ndarray, config: Optional[Any] = None) -> ResultProtocol:
        ...


@dataclass
class RawDataset(Generic[ProblemT, ResultT]):
    problems: List[ProblemT]
    results: List[ResultT]
    init_solution: np.ndarray
