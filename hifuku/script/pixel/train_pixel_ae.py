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
from hifuku.neuralnet import AutoEncoderConfig, PixelAutoEncoder
from hifuku.pool import TrivialProblemPool
from hifuku.rpbench_wrap import BubblyMeshPointConnectTask


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
    args = parser.parse_args()
    use_cache: bool = args.cache

    path = get_project_path("BubblyMeshPointConnectTask-AutoEncoder")
    path.mkdir(exist_ok=True, parents=True)
    cache_path = path / "mat_list_cache.pkl"

    if use_cache:
        with cache_path.open(mode="rb") as f:
            mat_list = pickle.load(f)
    else:
        pool = TrivialProblemPool(BubblyMeshPointConnectTask, 1).as_predicated()
        sampler = MultiProcessBatchProblemSampler()
        problems = sampler.sample_batch(100000, pool)

        mat_list = []
        for prob in tqdm.tqdm(problems):
            prob: BubblyMeshPointConnectTask
            gridsdf = prob._gridsdf
            assert gridsdf is not None
            mat = np.expand_dims(gridsdf.values.reshape(gridsdf.grid.sizes).T, axis=0)
            mat_list.append(mat)

        with cache_path.open(mode="wb") as f:
            pickle.dump(mat_list, f)

    dataset = MyDataset(mat_list)
    model = PixelAutoEncoder(AutoEncoderConfig())
    tcache = TrainCache.from_model(model)
    config = TrainConfig(n_epoch=300)
    train(path, tcache, dataset)
