from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple, Type, TypeVar

import numpy as np
import torch
from llazy.dataset import ChunkBase

ProblemT = TypeVar("ProblemT", bound="ProblemInterface")
ResultT = TypeVar("ResultT", bound="ResultProtocol")


class ResultProtocol(Protocol):
    nit: int
    success: bool


class ProblemInterface(ABC):
    @classmethod
    @abstractmethod
    def sample(cls: Type[ProblemT], n_sample: int) -> ProblemT:
        ...

    @abstractmethod
    def solve(
        self, sol_init: np.ndarray, config: Optional[Any] = None
    ) -> Tuple[ResultProtocol, ...]:
        ...

    @abstractmethod
    def get_mesh(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_descriptions(self) -> List[np.ndarray]:
        ...


@dataclass
class RawData(ChunkBase):
    mesh: np.ndarray
    descriptions: List[np.ndarray]
    nits: List[int]

    @classmethod
    def create(cls, problem: ProblemInterface, results: Tuple[ResultProtocol, ...]):
        mesh = problem.get_mesh()
        descriptions = problem.get_descriptions()
        nits = [result.nit for result in results]
        return cls(mesh, descriptions, nits)

    def dump_impl(self, path: Path) -> None:
        assert path.name.endswith(".npz")
        table = {}
        table["mesh"] = self.mesh
        table["description"] = np.array(self.descriptions)
        table["nit"] = np.array(self.nits)
        np.savez(str(path), **table)

    @classmethod
    def load(cls, path: Path) -> "RawData":
        kwargs = np.load(path)
        return cls(**kwargs)

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert False
        mesh = torch.from_numpy(self.mesh).float()
        description = torch.from_numpy(self.descriptions).float()
        nit = torch.tensor(self.nits, dtype=torch.float32)
        return mesh, description, nit

    def __len__(self) -> int:
        return 1
