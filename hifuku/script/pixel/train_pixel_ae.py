import argparse
import pickle
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import tqdm
from mohou.file import get_project_path
from mohou.trainer import TrainCache, TrainConfig, train
from torch.utils.data import Dataset

from hifuku.datagen import MultiProcessBatchProblemSampler
from hifuku.domain import select_domain
from hifuku.neuralnet import AutoEncoderConfig, PixelAutoEncoder
from hifuku.pool import TrivialProblemPool


@dataclass
class MyDataset(Dataset):
    mat_list: List[np.ndarray]

    def __len__(self) -> int:
        return len(self.mat_list)

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.from_numpy(self.mat_list[idx]).float()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="use cache")
    parser.add_argument("-domain", type=str, help="domain name")
    args = parser.parse_args()
    use_cache: bool = args.cache
    domain_name: str = args.domain

    domain = select_domain(domain_name)
    task_type = domain.task_type
    world_type = task_type.get_world_type()

    autoencoder_project_name = world_type.__name__ + "-AutoEncoder"
    path = get_project_path(autoencoder_project_name)
    path.mkdir(exist_ok=True, parents=True)
    cache_path = path / "mat_list_cache.pkl"

    if use_cache:
        with cache_path.open(mode="rb") as rf:
            mat_list = pickle.load(rf)
    else:
        pool = TrivialProblemPool(task_type, 1).as_predicated()
        sampler = MultiProcessBatchProblemSampler()  # type: ignore[var-annotated]
        problems = sampler.sample_batch(100000, pool)

        mat_list = []
        for prob in tqdm.tqdm(problems):
            gridsdf = prob.cache
            mat = np.expand_dims(gridsdf.values.reshape(gridsdf.grid.sizes).T, axis=0)
            mat_list.append(mat)

        with cache_path.open(mode="wb") as wf:
            pickle.dump(mat_list, wf)

    dataset = MyDataset(mat_list)
    model = PixelAutoEncoder(AutoEncoderConfig())
    tcache = TrainCache.from_model(model)
    config = TrainConfig(n_epoch=300)
    train(path, tcache, dataset)
