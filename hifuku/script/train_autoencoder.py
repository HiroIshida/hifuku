import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import List, Type

import numpy as np
import torch
import tqdm
from mohou.file import get_project_path
from mohou.trainer import TrainCache, TrainConfig, train
from rpbench.interface import TaskBase
from torch.utils.data import Dataset

from hifuku.datagen.utils import split_number
from hifuku.domain import select_domain
from hifuku.neuralnet import AutoEncoderConfig
from hifuku.script_utils import create_default_logger


def initialize(task_type: Type[TaskBase]):
    unique_seed = datetime.now().microsecond + os.getpid()
    np.random.seed(unique_seed)
    print("pid {}: random seed is set to {}".format(os.getpid(), unique_seed))
    global _task_type
    _task_type = task_type


def create_matrix(_):
    # should be common interface
    global _task_type
    sample = _task_type.sample()
    exp = sample.export_task_expression(use_matrix=True)
    mat = exp.get_matrix()
    has_channel = mat.ndim == 3
    if not has_channel:
        mat = np.expand_dims(mat, 0)
    return mat


@dataclass
class MyDataset(Dataset):
    mat_list: List[np.ndarray]

    def __len__(self) -> int:
        return len(self.mat_list)

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.from_numpy(self.mat_list[idx]).float()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="pr2_minifridge_sqp", help="")
    parser.add_argument("--sample", type=int, default=50000, help="")
    args = parser.parse_args()
    domain = select_domain(args.domain)
    task_type = domain.task_type
    task = task_type.sample()
    exp = task.export_task_expression(use_matrix=True)
    mat = exp.get_matrix()
    assert mat is not None

    has_channel = mat.ndim == 3
    if has_channel:
        n_channel = mat.shape[0]
        n_grid = mat.shape[1]
    else:
        n_channel = 1
        n_grid = mat.shape[0]

    path = get_project_path(domain.auto_encoder_project_name)
    path.mkdir(exist_ok=True, parents=True)
    logger = create_default_logger(path, "train_height_autoencoder")
    logger.info("create dateset from scratch")
    n_sample = args.sample

    # Somehow, if process all the sample at once with ProcessPoolExecutor, as the iteration goes on
    # processing starts to slow down. So, I split the number of samples into 10 parts and process them
    n_sample_list = split_number(n_sample, 10)
    assert sum(n_sample_list) == n_sample

    mat_list = []
    for i, n in enumerate(n_sample_list):
        dummy_args = [None] * n
        with ProcessPoolExecutor(20, initializer=initialize, initargs=(task_type,)) as pool:
            mat_list_part = list(tqdm.tqdm(pool.map(create_matrix, dummy_args), total=n))
        mat_list.extend(mat_list_part)

    dataset = MyDataset(mat_list)
    model = domain.auto_encoder_type(AutoEncoderConfig(n_grid=n_grid, n_channel=n_channel))
    # model = PixelAutoEncoder(AutoEncoderConfig(n_grid=n_grid, n_channel=n_channel))

    tcache = TrainCache.from_model(model)
    config = TrainConfig(n_epoch=10000)
    train(path, tcache, dataset, early_stopping_patience=100, num_workers=6)
