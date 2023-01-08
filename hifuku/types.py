import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, TypeVar

import numpy as np
import torch
from rpbench.interface import DescriptionTable
from skmp.solver.interface import ConfigProtocol, ResultProtocol
from skmp.trajectory import Trajectory

from hifuku.llazy.dataset import TensorChunkBase

ResultT = TypeVar("ResultT", bound=ResultProtocol)


@dataclass
class RawData(TensorChunkBase):
    init_solution: Trajectory
    desc: DescriptionTable
    results: Tuple[ResultProtocol, ...]
    solver_config: ConfigProtocol

    def dump_impl(self, path: Path) -> None:
        assert path.name.endswith(".pkl")
        with path.open(mode="wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path, decompress: bool = False) -> "RawData":
        if decompress:
            assert path.name.endswith(".gz")
            subprocess.run("gunzip {} --keep --force".format(path), shell=True)
            path.name
            path = path.parent / path.stem

        assert path.name.endswith(".pkl")
        with path.open(mode="rb") as f:
            loaded = pickle.load(f)
        return loaded

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        np_mesh = self.desc.get_mesh()
        torch_mesh = torch.from_numpy(np_mesh).float().unsqueeze(0)

        np_vector_descs = self.desc.get_vector_descs()

        if len(np_vector_descs) == 0:
            # FIXME: when wcd len == 0, wd_1dim_desc_tensor is ignored ...?
            torch_wcd_descs = torch.zeros(0, 0).float()
            torch_nits = torch.zeros(0, 0).float()
            return torch_mesh, torch_wcd_descs, torch_nits
        else:
            torch_vector_descs = [torch.from_numpy(desc).float() for desc in np_vector_descs]
            torch_wcd_descs = torch.stack(torch_vector_descs)

            def get_clamped_iter(result: ResultProtocol) -> int:
                if result.traj is None:
                    return self.solver_config.n_max_call + 1
                return result.n_call

            nits = np.array([get_clamped_iter(r) for r in self.results])
            torch_nits = torch.from_numpy(nits).float()
            return torch_mesh, torch_wcd_descs, torch_nits
