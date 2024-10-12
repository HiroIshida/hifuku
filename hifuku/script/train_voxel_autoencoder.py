import datetime
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List

import numpy as np
import threadpoolctl
import torch
import tqdm
from mohou.file import get_project_path
from mohou.trainer import TrainCache, TrainConfig, train
from rpbench.articulated.world.jail import JailWorld
from torch.utils.data import Dataset

from hifuku.neuralnet import VoxelAutoEncoder, VoxelAutoEncoderConfig
from hifuku.script_utils import create_default_logger


def setup_worker():
    pid = os.getpid()
    now = datetime.datetime.now()
    seed = pid + now.second + now.microsecond
    np.random.seed(seed)


def sample_task_param(_):
    return JailWorld.sample().serialize()


@dataclass
class MyDataset(Dataset):
    task_params: List[bytes]

    def __init__(self, n_sample: int):
        with ProcessPoolExecutor(10, initializer=setup_worker) as executor:
            self.task_params = list(
                tqdm.tqdm(
                    executor.map(sample_task_param, range(n_sample)),
                    total=n_sample,
                    desc="Sampling task parameters",
                )
            )
        # check hash and check that no overlap
        hashes = set()
        for param in self.task_params:
            h = hashlib.sha256(param).digest()
            assert h not in hashes
            hashes.add(h)

    def __len__(self) -> int:
        return len(self.task_params)

    def __getitem__(self, idx) -> torch.Tensor:
        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            param = self.task_params[idx]
            world = JailWorld.deserialize(param)
            vmap = world.voxels.to_3darray()
            vmap = vmap[np.newaxis, :]
        return torch.from_numpy(vmap).float()


if __name__ == "__main__":
    dataset = MyDataset(n_sample=3000)
    pp = get_project_path("unko")
    model = VoxelAutoEncoder(VoxelAutoEncoderConfig(n_grid=56, output_binary=True))
    dummy_input = torch.randn(1, 1, 56, 56, 56)
    out = model(dummy_input)

    tcache = TrainCache.from_model(model)
    tconfig = TrainConfig(n_epoch=10000)
    logger = create_default_logger(pp, "train_voxel_autoencoder")
    train(pp, tcache, dataset, early_stopping_patience=100, num_workers=6)
