import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import torch
import tqdm
from mohou.file import get_project_path
from mohou.trainer import TrainCache, TrainConfig, train
from rpbench.articulated.jaxon.below_table import BelowTableClutteredWorld
from rpbench.articulated.world.ground import GroundClutteredWorld
from rpbench.articulated.world.shelf import ShelfBoxClutteredWorld
from rpbench.two_dimensional.bubbly_world import (
    BubblyWorldComplex,
    BubblyWorldEmpty,
    BubblyWorldSimple,
)
from torch.utils.data import Dataset

from hifuku.neuralnet import AutoEncoderConfig, PixelAutoEncoder
from hifuku.script_utils import create_default_logger


def initialize(world_type):
    unique_seed = datetime.now().microsecond + os.getpid()
    np.random.seed(unique_seed)
    print("pid {}: random seed is set to {}".format(os.getpid(), unique_seed))
    global _world_type
    _world_type = world_type


def create_mesh(_):
    # should be common interface
    global _world_type
    sample = _world_type.sample()
    if hasattr(sample, "heightmap"):
        mesh = sample.heightmap()
    elif hasattr(sample, "get_grid_map"):
        mesh = sample.get_grid_map()
    else:
        assert False
    ident = np.random.randint(10**10)
    return np.expand_dims(mesh, axis=0), ident


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
    parser.add_argument("-world", type=str, help="world name", default="bubbly_simple")
    parser.add_argument("-n", type=int, help="number of batch", default=20000)
    parser.add_argument("-m", type=int, help="number of iter", default=5)

    args = parser.parse_args()
    use_cache: bool = args.cache

    if args.world == "gc":
        world_type = GroundClutteredWorld
    elif args.world == "shelf":
        world_type = ShelfBoxClutteredWorld
    elif args.world == "below":
        world_type = BelowTableClutteredWorld
    elif args.world == "bubbly_simple":
        world_type = BubblyWorldSimple
    elif args.world == "bubbly_complex":
        world_type = BubblyWorldComplex
    elif args.world == "bubbly_empty":
        world_type = BubblyWorldEmpty
    elif args.world == "bubbly_empty":
        world_type = BubblyWorldEmpty
    else:
        assert False

    autoencoder_project_name = world_type.__name__ + "-AutoEncoder"
    path = get_project_path(autoencoder_project_name)
    path.mkdir(exist_ok=True, parents=True)

    logger = create_default_logger(path, "train_height_autoencoder")

    cache_path = path / "mat_list_cache.pkl"

    if use_cache:
        logger.info("use cache")
        with cache_path.open(mode="rb") as rf:
            mat_list = pickle.load(rf)
    else:
        logger.info("create dateset from scratch")
        mat_list = []
        idents = []
        n_sample_per = args.n
        for _ in tqdm.tqdm(range(args.m)):
            with ProcessPoolExecutor(
                22, initializer=initialize, initargs=(world_type,)
            ) as executor:
                for mesh, ident in tqdm.tqdm(
                    executor.map(create_mesh, range(n_sample_per)), total=n_sample_per
                ):
                    mat_list.append(mesh)
                    idents.append(ident)
            collision_rate = 1.0 - float(len(set(idents))) / len(idents)
            print("current collision rate: {}".format(collision_rate))

        with cache_path.open(mode="wb") as wf:
            pickle.dump(mat_list, wf)

    sample = world_type.sample(True)
    if hasattr(sample, "heightmap"):
        grid_map = sample.heightmap()
    elif hasattr(sample, "get_grid_map"):
        grid_map = sample.get_grid_map()
    else:
        assert False
    n_grid = grid_map.shape[0]
    assert n_grid in [56, 112]

    dataset = MyDataset(mat_list)
    model = PixelAutoEncoder(AutoEncoderConfig(n_grid=n_grid))
    tcache = TrainCache.from_model(model)
    config = TrainConfig(n_epoch=300)
    train(path, tcache, dataset)
