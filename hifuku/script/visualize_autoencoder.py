import argparse
from typing import Literal

import matplotlib.pyplot as plt
import torch

from hifuku.domain import select_domain
from hifuku.neuralnet import NeuralAutoEncoderBase
from hifuku.script_utils import load_compatible_autoencoder

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

    n_grid: Literal[56, 112] = mat.shape[0]  # type: ignore

    ae: NeuralAutoEncoderBase = load_compatible_autoencoder(domain, True, n_grid=n_grid)  # type: ignore
    print(ae)

    for _ in range(10):
        sample = task_type.sample()
        exp = sample.export_task_expression(use_matrix=True)
        mat = exp.get_matrix()
        has_channel = len(mat.shape) == 3
        if has_channel:
            matten = torch.from_numpy(mat).float().unsqueeze(0)
        else:
            matten = torch.from_numpy(mat).float().unsqueeze(0).unsqueeze(0)
        matten = matten.to(ae.device)
        matten_reconst = ae.decoder(ae.encoder(matten))
        target_channel = 1
        if has_channel:
            mat_reconst = matten_reconst.squeeze(0)[target_channel].cpu().detach().numpy()
        else:
            mat_reconst = matten_reconst.squeeze(0).squeeze(0).cpu().detach().numpy()
        # compare original and reconstructed side by side
        fig, ax = plt.subplots(1, 2)
        if has_channel:
            ax[0].imshow(mat[target_channel])
        else:
            ax[0].imshow(mat)
        ax[1].imshow(mat_reconst)
        plt.show()
