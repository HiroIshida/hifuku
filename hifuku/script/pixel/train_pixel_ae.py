import argparse
import pickle
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import tqdm
from mohou.file import get_project_path
from mohou.script_utils import create_default_logger
from mohou.trainer import TrainCache, TrainConfig, train
from torch.utils.data import Dataset

from hifuku.datagen import MultiProcessBatchTaskSampler
from hifuku.domain import select_domain
from hifuku.neuralnet import AutoEncoderConfig, PixelAutoEncoder
from hifuku.pool import TrivialTaskPool


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
    parser.add_argument("--domain", type=str, help="domain name")
    parser.add_argument("--sample", type=int, help="sample size", default=20000)
    parser.add_argument("--bottleneck", type=int, help="bottleneck size", default=200)
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

    logger = create_default_logger(path, "pixel_ae")

    if use_cache:
        with cache_path.open(mode="rb") as rf:
            mat_list = pickle.load(rf)
    else:
        pool = TrivialTaskPool(task_type, 1).as_predicated()
        sampler = MultiProcessBatchTaskSampler()  # type: ignore[var-annotated]
        problems = sampler.sample_batch(args.sample, pool)

        mat_list = []
        for prob in tqdm.tqdm(problems):
            # gridsdf = prob.cache
            # mat = np.expand_dims(gridsdf.values.reshape(gridsdf.grid.sizes).T, axis=0)
            mat = np.expand_dims(prob.world.heightmap(), axis=0)
            mat_list.append(mat)

        with cache_path.open(mode="wb") as wf:
            pickle.dump(mat_list, wf)

    dataset = MyDataset(mat_list)
    model = PixelAutoEncoder(AutoEncoderConfig(n_grid=56, dim_bottleneck=args.bottleneck))
    tcache = TrainCache.from_model(model)
    config = TrainConfig(n_epoch=2000)
    train(path, tcache, dataset)
