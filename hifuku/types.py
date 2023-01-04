import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, TypeVar

import numpy as np
import torch
from rpbench.interface import DescriptionTable, TaskBase
from skmp.solver.interface import ConfigProtocol, ResultProtocol

from hifuku.llazy.dataset import ChunkBase

ResultT = TypeVar("ResultT", bound=ResultProtocol)


@dataclass
class RawData(ChunkBase):
    desc: DescriptionTable
    results: Tuple[ResultProtocol]
    solver_config: ConfigProtocol

    @classmethod
    def construct(
        cls, task: TaskBase, results: Tuple[ResultProtocol], solcon: ConfigProtocol
    ) -> "RawData":
        desc = task.export_table()
        return cls(desc, results, solcon)

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
        # currently we assume that world and world-conditioned descriptions follows
        # the following rule.
        # wd refere to world description and wcd refere to world-condtioned descriotion
        # - wd must have either 2dim or 3dim array and not both
        # - wd has only one key for 2dim or 3dim array
        # - wd may have 1 dim array
        # - wcd does not have 2dim or 3dim array
        # - wcd must have 1dim array
        # - wcd may be empty, but wd must not be empty
        # to remove the above limitation, create a task class which inherit rpbench Task
        # which is equipped with desc-2-tensor-tuple conversion rule

        # parse world description
        wd_ndim_to_value = {v.ndim: v for v in self.desc.world_desc_dict.values()}
        ndim_set = wd_ndim_to_value.keys()

        contains_either_2or3_not_both = (2 in ndim_set) ^ (3 in ndim_set)
        assert contains_either_2or3_not_both
        if 2 in ndim_set:
            mesh = wd_ndim_to_value[2]
        elif 3 in ndim_set:
            mesh = wd_ndim_to_value[3]
        else:
            assert False
        torch_mesh = torch.from_numpy(mesh).float().unsqueeze(dim=0)

        one_key_per_dim = len(wd_ndim_to_value) == len(self.desc.world_desc_dict)
        assert one_key_per_dim
        wd_1dim_desc: Optional[np.ndarray] = None
        if 1 in ndim_set:
            wd_1dim_desc = wd_ndim_to_value[1]

        # parse world-conditioned description

        if len(self.desc.wcond_desc_dicts) == 0:
            # FIXME: when wcd len == 0, wd_1dim_desc_tensor is ignored ...?
            torch_wcd_descs = torch.zeros(0, 0).float()
            torch_nits = torch.zeros(0, 0).float()
            return torch_mesh, torch_wcd_descs, torch_nits
        else:
            wcd_desc_dict = self.desc.wcond_desc_dicts[0]
            ndims = set([v.ndim for v in wcd_desc_dict.values()])
            assert ndims == {1}

            torch_wcd_desc_list = []
            len(self.desc.wcond_desc_dicts)
            for wcd_desc_dict in self.desc.wcond_desc_dicts:
                wcd_desc_vec_list = []
                if wd_1dim_desc is not None:
                    wcd_desc_vec_list.append(wd_1dim_desc)
                wcd_desc_vec_list.extend(list(wcd_desc_dict.values()))
                wcd_desc_vec_cat = np.concatenate(wcd_desc_vec_list)
                torch_desc = torch.from_numpy(wcd_desc_vec_cat).float()
                torch_wcd_desc_list.append(torch_desc)
            torch_wcd_descs = torch.stack(torch_wcd_desc_list)

            nits = np.array([r.n_call for r in self.results])
            torch_nits = torch.from_numpy(nits).float()
            return torch_mesh, torch_wcd_descs, torch_nits
