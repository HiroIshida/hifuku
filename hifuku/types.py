from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, Protocol, Tuple, Type, TypeVar

import numpy as np
import torch
from llazy.dataset import PicklableChunkBase

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

    @abstractmethod
    def get_mesh(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_description(self) -> np.ndarray:
        ...


@dataclass
class RawData(Generic[ProblemT, ResultT], PicklableChunkBase):
    problem: ProblemT
    result: ResultT

    def __len__(self) -> int:
        return 1

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mesh = torch.from_numpy(self.problem.get_mesh()).float()
        description = torch.from_numpy(self.problem.get_description()).float()
        nit = torch.tensor(self.result.nit, dtype=torch.float32)
        return mesh, description, nit
