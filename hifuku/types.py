import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type, TypeVar, Union

import numpy as np
import torch

from hifuku.llazy.dataset import ChunkBase

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
    class SamplingBasedInitialguessFail(Exception):
        pass

    @classmethod
    @abstractmethod
    def sample(cls: Type[ProblemT], n_sample: int) -> ProblemT:
        ...

    @classmethod
    @abstractmethod
    def create_standard(cls: Type[ProblemT]) -> ProblemT:
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

    @abstractmethod
    def n_problem(self) -> int:
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
    init_solution: np.ndarray
    maxiter: int
    maxfev: int

    def __post_init__(self):
        assert len(self.descriptions) == len(self.nfevs)

    @classmethod
    def create(
        cls,
        problem: ProblemInterface,
        results: Tuple[ResultProtocol, ...],
        init_solution: np.ndarray,
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
        return cls(
            mesh, descriptions, nits, nfevs, successes, solutions, init_solution, maxiter, maxfev
        )

    def dump_impl(self, path: Path) -> None:
        assert path.name.endswith(".npz")
        # dump as npz rather than pkl for future data backward compatibility
        table: Dict[str, Union[np.ndarray, int]] = {}
        table["mesh"] = self.mesh
        table["descriptions"] = np.array(self.descriptions)
        table["nits"] = np.array(self.nits, dtype=int)
        table["nfevs"] = np.array(self.nfevs, dtype=int)
        table["successes"] = np.array(self.successes, dtype=bool)
        table["solutions"] = np.array(self.solutions)
        table["init_solution"] = np.array(self.init_solution)
        table["maxiter"] = self.maxiter
        table["maxfev"] = self.maxfev

        for field in fields(self):
            assert field.name in table

        np.savez(str(path), **table)

    @classmethod
    def load(cls, path: Path, decompress: bool = False) -> "RawData":
        path_original = path
        if decompress:
            assert path.name.endswith(".gz")
            subprocess.run("gunzip {} --keep --force".format(path), shell=True)
            path.name
            path = path.parent / path.stem

        assert path.name.endswith(".npz")
        loaded = np.load(path)

        if path_original.name.endswith(".gz"):
            os.remove(path)
        assert path_original.exists()

        kwargs = {}
        kwargs["mesh"] = loaded["mesh"]
        kwargs["descriptions"] = list(loaded["descriptions"])
        kwargs["nits"] = [int(e) for e in loaded["nits"]]
        kwargs["nfevs"] = [int(e) for e in loaded["nfevs"]]
        kwargs["successes"] = list([bool(e) for e in loaded["successes"]])
        kwargs["solutions"] = list(loaded["solutions"])
        kwargs["init_solution"] = loaded["init_solution"]
        kwargs["maxiter"] = int(loaded["maxiter"].item())
        kwargs["maxfev"] = int(loaded["maxfev"].item())

        for field in fields(cls):
            assert field.name in kwargs

        return cls(**kwargs)

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mesh = torch.from_numpy(self.mesh).float().unsqueeze(dim=0)
        descriptions_np = np.stack(self.descriptions)
        description = torch.from_numpy(descriptions_np).float()

        crop_nfev = self.maxfev
        nfevs_np = np.minimum(
            np.array(self.nfevs) + np.array(np.logical_not(self.successes, dtype=bool)) * crop_nfev,
            crop_nfev,
        )
        nfevs = torch.from_numpy(nfevs_np).float()

        solution_np = np.array(self.solutions)
        solutions = torch.from_numpy(solution_np).float()
        return mesh, description, nfevs, solutions

    def __len__(self) -> int:
        return 1
