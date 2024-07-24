from dataclasses import dataclass

import numpy as np
import threadpoolctl
import torch
from mohou.file import get_project_path
from mohou.trainer import TrainCache, TrainConfig, train
from rpbench.articulated.pr2.minifridge import PR2MiniFridgeTask
from torch.utils.data import Dataset

from hifuku.domain import PR2MiniFridgeVoxel_RRT2000
from hifuku.neuralnet import VoxelAutoEncoder, VoxelAutoEncoderConfig
from hifuku.pool import TaskPool
from hifuku.script_utils import create_default_logger


@dataclass
class MyDataset(Dataset):
    task_params: np.ndarray

    def __init__(self, n_sample: int, distributed: bool = False):
        # sample tasks beforehand as it is too time-consuming to be done in the training loop
        domain = PR2MiniFridgeVoxel_RRT2000
        if distributed:
            sampler = domain.get_distributed_batch_sampler()
        else:
            sampler = domain.get_multiprocess_batch_sampler()
        pool = TaskPool(domain.task_type)
        task_params = sampler.sample_batch(n_sample, pool)
        self.task_params = task_params

    def __len__(self) -> int:
        return len(self.task_params)

    def __getitem__(self, idx) -> torch.Tensor:
        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            param = self.task_params[idx]
            task = PR2MiniFridgeTask.from_task_param(param)
            vmap = task.world.create_voxelmap()
            vmap = vmap[np.newaxis, :]
        return torch.from_numpy(vmap).float()


if __name__ == "__main__":
    dataset = MyDataset(n_sample=50000, distributed=True)
    domain = PR2MiniFridgeVoxel_RRT2000
    pp = get_project_path(domain.auto_encoder_project_name)
    warm = True
    if warm:
        tcache = TrainCache.load_latest(pp, VoxelAutoEncoder)
        tconfig = TrainConfig(n_epoch=150)
    else:
        model = VoxelAutoEncoder(VoxelAutoEncoderConfig())
        tcache = TrainCache.from_model(model)
        tconfig = TrainConfig(n_epoch=300)
    logger = create_default_logger(pp, "train_voxel_autoencoder")
    train(pp, tcache, dataset, tconfig, num_workers=10)
