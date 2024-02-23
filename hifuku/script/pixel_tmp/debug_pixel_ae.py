import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from mohou.file import get_project_path
from mohou.trainer import TrainCache
from rpbench.two_dimensional.bubbly_world import BubblyComplexMeshPointConnectTask

from hifuku.neuralnet import PixelAutoEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int)
    args = parser.parse_args()
    np.random.seed(args.seed)

    # path = get_project_path("GroundClutteredWorld-AutoEncoder")
    # path = get_project_path("TabletopBoxWorld-AutoEncoder")
    # path = get_project_path("BelowTableClutteredWorld-AutoEncoder")
    # path = get_project_path("TabletopClutteredOvenWorld-AutoEncoder")
    # path = get_project_path(BubblyWorldSimple - AutoEncoder)
    # path = get_project_path("TabletopClutteredFridgeWorld-AutoEncoder")
    # path = get_project_path("TabletopClutteredFridgeWorld-AutoEncoder")
    path = get_project_path("BubblyWorldComplex-AutoEncoder")
    ae_model = TrainCache.load_latest(path, PixelAutoEncoder).best_model

    # task = TabletopClutteredOvenReachingTask
    # task_type = TabletopClutteredFridgeReachingTask
    task_type = BubblyComplexMeshPointConnectTask
    world_type = task_type.get_world_type()

    # pool = TrivialProblemPool(HumanoidGroundRarmReachingTask, 1).as_predicated()
    # sampler = MultiProcessBatchProblemSampler()  # type: ignore[var-annotated]
    # problems = sampler.sample_batch(100, pool)

    mat_list = []
    for _ in range(1):
        task = task_type.sample(1)
        # viewer = task.create_viewer(mode="static")
        # viewer.save_image(f"./world-{args.seed}.png")

        # hmap = ShelfBoxClutteredWorld.sample().heightmap()
        # hmap = task.world.heightmap()
        hmap = task.world.get_grid_map()
        mat_list.append(np.expand_dims(hmap, axis=0))

    mat_data = torch.from_numpy(np.array(mat_list)).float()
    mat_data_reconst = ae_model.decoder((ae_model.encode(mat_data)))
    mat_reconst_list = mat_data_reconst.detach().cpu().numpy()

    for mat, mat_reconst in zip(mat_list, mat_reconst_list):
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(mat[0])
        axes[1].imshow(mat_reconst[0])
        plt.savefig(f"./hmap-{args.seed}.png")
        # plt.show()
