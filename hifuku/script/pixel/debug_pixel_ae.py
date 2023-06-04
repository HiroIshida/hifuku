import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from mohou.file import get_project_path
from mohou.trainer import TrainCache

from hifuku.datagen import MultiProcessBatchProblemSampler
from hifuku.neuralnet import PixelAutoEncoder
from hifuku.pool import TrivialProblemPool
from hifuku.rpbench_wrap import BubblySimpleMeshPointConnectTask

if __name__ == "__main__":
    path = get_project_path("BubblyMeshPointConnectTask-AutoEncoder")
    ae_model = TrainCache.load_latest(path, PixelAutoEncoder).best_model

    pool = TrivialProblemPool(BubblySimpleMeshPointConnectTask, 1).as_predicated()
    sampler = MultiProcessBatchProblemSampler()  # type: ignore[var-annotated]
    problems = sampler.sample_batch(100, pool)

    mat_list = []
    for prob in tqdm.tqdm(problems):
        gridsdf = prob._gridsdf
        assert gridsdf is not None
        mat = np.expand_dims(gridsdf.values.reshape(gridsdf.grid.sizes).T, axis=0)
        mat_list.append(mat)

    mat_data = torch.from_numpy(np.array(mat_list)).float()
    mat_data_reconst = ae_model.decoder((ae_model.encode(mat_data)))
    mat_reconst_list = mat_data_reconst.detach().cpu().numpy()

    for mat, mat_reconst in zip(mat_list, mat_reconst_list):
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(mat[0])
        axes[1].imshow(mat_reconst[0])
        plt.show()
