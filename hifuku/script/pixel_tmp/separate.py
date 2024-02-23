import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import tqdm
from mohou.file import get_project_path
from mohou.trainer import TrainCache
from rpbench.articulated.world.ground import GroundClutteredWorld
from sklearn.decomposition import PCA

from hifuku.neuralnet import PixelAutoEncoder


def init():
    unique_seed = datetime.now().microsecond + os.getpid()
    np.random.seed(unique_seed)
    print("pid {}: random seed is set to {}".format(os.getpid(), unique_seed))


def work(_):
    world = GroundClutteredWorld.sample(False)
    hmap = world.create_exact_heightmap()
    mat = np.expand_dims(hmap, axis=0)
    return mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("-t", type=int, default=10)
    args = parser.parse_args()
    n_point = args.n
    path = get_project_path("GroundClutteredWorld-AutoEncoder")
    cache_path = Path("/tmp/debug_ae-{}.pkl".format(n_point))
    if not cache_path.exists():
        print("{} not found so create".format(cache_path))
        mat_list = []
        with ProcessPoolExecutor(12, initializer=init) as executor:
            for e in tqdm.tqdm(executor.map(work, range(n_point)), total=n_point):
                mat_list.append(e)
        ae_model = TrainCache.load_latest(path, PixelAutoEncoder).best_model
        mat_data = torch.from_numpy(np.array(mat_list)).float()
        encoded = ae_model.encode(mat_data).detach().cpu().numpy()

        pca_model = PCA(n_components=2)
        pca_encoded = pca_model.fit_transform(encoded)

        with cache_path.open(mode="wb") as f:
            pickle.dump(pca_encoded, f)
        print("saved to {}".format(cache_path))

    with cache_path.open(mode="rb") as f:
        pca_encoded = pickle.load(f)
    print("loaded from {}".format(cache_path))

    G = nx.Graph()
    for i in range(len(pca_encoded)):
        G.add_node(i)
    threshold = args.t
    for i in range(n_point):
        for j in range(i + 1, n_point):
            dist = np.linalg.norm(pca_encoded[i] - pca_encoded[j])
            if dist < threshold:
                G.add_edge(i, j)
    print("construcetd graph")
    nx.draw(G, with_labels=False, node_size=10, width=0.4)
    plt.show()

    # distances = pairwise_distances(pca_encoded)
    # print(distances)
    # print(sum(pca_encoded[:, 0] > 100))
    # print(pca_encoded.shape)
    # print(pca_encoded)
    # plt.scatter(pca_encoded[:, 0], pca_encoded[:, 1], s=50, cmap='viridis')
    # plt.show()
