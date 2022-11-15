from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from llazy.dataset import ChunkBase

ProblemT = TypeVar("ProblemT", bound="ProblemInterface")
ResultT = TypeVar("ResultT", bound="ResultProtocol")


class ResultProtocol(Protocol):
    nit: int
    nfev: int
    success: bool
    x: np.ndarray


class SolverConfigProtocol(Protocol):
    maxiter: int
    maxfev: int


class ProblemInterface(ABC):
    @classmethod
    @abstractmethod
    def sample(cls: Type[ProblemT], n_sample: int) -> ProblemT:
        ...

    @abstractmethod
    def solve(
        self, sol_init: Optional[np.ndarray] = None, config: Optional[Any] = None
    ) -> Tuple[ResultProtocol, ...]:
        ...

    @abstractmethod
    def get_mesh(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_descriptions(self) -> List[np.ndarray]:
        ...


@dataclass
class RawMeshData(ChunkBase):
    mesh: np.ndarray

    def dump_impl(self, path: Path) -> None:
        assert path.name.endswith(".npz")
        table = {}
        table["mesh"] = self.mesh
        np.savez(str(path), **table)

    @classmethod
    def load(cls, path: Path) -> "RawMeshData":
        loaded = np.load(path)
        kwargs = {}
        kwargs["mesh"] = loaded["mesh"]
        return cls(**kwargs)

    def to_tensors(self) -> Tuple[torch.Tensor]:
        mesh = torch.from_numpy(self.mesh).float().unsqueeze(dim=0)
        return (mesh,)

    def __len__(self) -> int:
        return 1


@dataclass
class RawData(ChunkBase):
    mesh: np.ndarray
    descriptions: List[np.ndarray]
    nits: List[int]
    nfevs: List[int]
    successes: List[bool]
    solutions: List[np.ndarray]
    maxiter: int
    maxfev: int

    def __post_init__(self):
        assert len(self.descriptions) == len(self.nfevs)

    @classmethod
    def create(
        cls,
        problem: ProblemInterface,
        results: Tuple[ResultProtocol, ...],
        config: SolverConfigProtocol,
    ):
        mesh = problem.get_mesh()
        descriptions = problem.get_descriptions()
        nits = [result.nit for result in results]
        nfevs = [result.nfev for result in results]
        successes = [result.success for result in results]
        solutions = [result.x for result in results]
        maxiter = config.maxiter
        maxfev = config.maxfev
        return cls(mesh, descriptions, nits, nfevs, successes, solutions, maxiter, maxfev)

    def dump_impl(self, path: Path) -> None:
        assert path.name.endswith(".npz")
        # dump as npz rather than pkl for future data backward compatibility
        table: Dict[str, Union[np.ndarray, int]] = {}
        table["mesh"] = self.mesh
        table["descriptions"] = np.array(self.descriptions)
        table["nits"] = np.array(self.nits)
        table["nfevs"] = np.array(self.nfevs)
        table["successes"] = np.array(self.successes)
        table["solutions"] = np.array(self.solutions)
        table["maxiter"] = self.maxiter
        table["maxfev"] = self.maxfev

        for field in fields(self):
            assert field.name in table

        np.savez(str(path), **table)

    @classmethod
    def load(cls, path: Path) -> "RawData":
        loaded = np.load(path)
        kwargs = {}
        kwargs["mesh"] = loaded["mesh"]
        kwargs["descriptions"] = list(loaded["descriptions"])
        kwargs["nits"] = list(loaded["nits"])
        kwargs["nfevs"] = list(loaded["nfevs"])
        kwargs["successes"] = list(loaded["successes"].astype(bool))
        kwargs["solutions"] = list(loaded["solutions"])
        kwargs["maxiter"] = loaded["maxiter"]
        kwargs["maxfev"] = loaded["maxfev"]

        for field in fields(cls):
            assert field.name in kwargs

        return cls(**kwargs)

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mesh = torch.from_numpy(self.mesh).float().unsqueeze(dim=0)
        descriptions_np = np.stack(self.descriptions)
        description = torch.from_numpy(descriptions_np).float()

        crop_nfev = self.maxfev
        nfevs_np = np.minimum(
            np.array(self.nfevs) + np.array(self.successes, dtype=bool) * crop_nfev, crop_nfev
        )
        nfevs = torch.from_numpy(nfevs_np).float()

        solution_np = np.array(self.solutions)
        solutions = torch.from_numpy(solution_np).float()
        return mesh, description, nfevs, solutions

    def __len__(self) -> int:
        return 1
