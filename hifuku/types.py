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

    def __post_init__(self):
        assert len(self.descriptions) == len(self.nits)

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
        table["descriptions"] = np.array(self.descriptions)
        table["nits"] = np.array(self.nits)
        np.savez(str(path), **table)

    @classmethod
    def load(cls, path: Path) -> "RawData":
        loaded = np.load(path)
        kwargs = {}
        kwargs["mesh"] = loaded["mesh"]
        kwargs["descriptions"] = list(loaded["descriptions"])
        kwargs["nits"] = list(loaded["nits"])
        return cls(**kwargs)

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_actual_problem = len(self.descriptions)

        mesh = torch.from_numpy(self.mesh).float().unsqueeze(dim=0)
        meshes = mesh.expand(n_actual_problem, -1, -1, -1)

        descriptions_np = np.stack(self.descriptions)
        description = torch.from_numpy(descriptions_np).float()
        nits = torch.tensor(self.nits, dtype=torch.float32)
        return meshes, description, nits

    def __len__(self) -> int:
        return 1
