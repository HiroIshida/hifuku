from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm
from llazy.dataset import LazyDecomplessDataLoader, LazyDecomplessDataset
from mohou.model.common import LossDict, ModelBase, ModelConfigBase
from mohou.utils import detect_device
from torch import Tensor
from torch.utils.data import Dataset

from hifuku.types import ProblemInterface, RawData


@dataclass
class IterationPredictorDataset(Dataset):
    meshe_encodeds: torch.Tensor
    descriptions: torch.Tensor
    itervals: torch.Tensor
    solutions: torch.Tensor

    def __len__(self) -> int:
        return len(self.meshe_encodeds)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        return (
            self.meshe_encodeds[idx],
            self.descriptions[idx],
            self.itervals[idx],
            self.solutions[idx],
        )

    @classmethod
    def load(cls, dataset_path: Path, ae_model: "VoxelAutoEncoder"):
        device = detect_device()
        ae_model.put_on_device(device)
        dataset = LazyDecomplessDataset.load(dataset_path, RawData, n_worker=-1)
        loader = LazyDecomplessDataLoader(dataset, batch_size=1000, shuffle=False)

        encoded_list = []
        descriptions_list = []
        itervals_list = []
        solutions_list = []

        # create minibatch list
        for sample in tqdm.tqdm(loader):
            mesh, description, iterval, solution = sample

            # process mesh (compress by autoenocoder)
            mesh = mesh.to(device)  # n_mini_batch x (*shape)
            encoded = ae_model.encoder(mesh)
            encoded_list.append(encoded.detach().cpu())

            # process others
            descriptions_list.append(description)
            itervals_list.append(iterval)
            solutions_list.append(solution)

        mesh_encodeds_concat = torch.cat(encoded_list, dim=0)  # n_batch x n_bottleneck
        descriptions_concat = torch.cat(descriptions_list, dim=0)
        itervals_concat = torch.cat(itervals_list, dim=0)
        solutsions_concat = torch.cat(solutions_list, dim=0)
        return cls(mesh_encodeds_concat, descriptions_concat, itervals_concat, solutsions_concat)


@dataclass
class IterationPredictorConfig(ModelConfigBase):
    dim_problem_descriptor: int
    dim_conv_bottleneck: int
    dim_solution: int
    dim_conv: int = 3
    dim_description_expand: int = 300
    use_solution_pred: bool = False


class IterationPredictor(ModelBase[IterationPredictorConfig]):
    encoder: nn.Sequential
    linears: nn.Sequential
    iter_linears: nn.Sequential
    solution_linears: Optional[nn.Sequential]
    description_expand_lineras: nn.Sequential
    margin: Optional[float] = None

    def _setup_from_config(self, config: IterationPredictorConfig) -> None:

        self.description_expand_lineras = nn.Sequential(
            nn.Linear(config.dim_problem_descriptor, config.dim_description_expand),
            nn.ReLU(),
            nn.Linear(config.dim_description_expand, config.dim_description_expand),
            nn.ReLU(),
        )

        n_input = config.dim_conv_bottleneck + config.dim_description_expand

        self.linears = nn.Sequential(
            nn.Linear(n_input, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
        )

        self.iter_linears = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 1))

        if config.use_solution_pred:
            self.solution_linears = nn.Sequential(
                nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, config.dim_solution)
            )
        else:
            self.solution_linears = None

    def forward(self, sample: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        mesh_features, descriptor = sample
        # descriptor: (n_batch x n_pose x dim_descriptor)
        n_batch, n_pose, _ = descriptor.shape

        # mesh_features_rep: (n_batch * n_pose x dim_feature)
        mesh_features_rep = mesh_features.repeat(n_pose, 1)

        # descriptor_flatten: (n_batch * n_pose x dim_descriptor)
        descriptor_flatten = descriptor.reshape(n_batch * n_pose, -1)
        expanded = self.description_expand_lineras(descriptor_flatten)

        vectors = torch.concat([mesh_features_rep, expanded], dim=1)
        tmp = self.linears(vectors)

        # iter_pred: (n_batch * n_pose)
        iter_pred = self.iter_linears(tmp)
        # iter_pred: (n_batch x n_pose)
        iter_pred = iter_pred.reshape(n_batch, n_pose)

        if self.solution_linears is not None:
            # solution_pred: (n_batch * n_pose x dim_solution)
            solution_pred = self.solution_linears(tmp)
            # solution_pred: (n_batch x n_pose x dim_solution)
            solution_pred = solution_pred.reshape(n_batch, n_pose, -1)
        else:
            solution_pred = None
        return iter_pred, solution_pred

    def loss(self, sample: Tuple[Tensor, Tensor, Tensor, Tensor]) -> LossDict:
        mesh_encoded, descriptor, iterval, solution = sample
        # mesh: (n_batch, (mesh_size))
        # descriptors: (n_batch, n_pose, (descriptor_dim))
        # iterval: (n_batch, n_pose, (,))
        dic = {}
        iter_pred, solution_pred = self.forward((mesh_encoded, descriptor))
        iter_loss = nn.MSELoss()(iter_pred, iterval)
        dic["iter"] = iter_loss

        if solution_pred is not None:
            solution_loss = nn.MSELoss()(solution_pred, solution) * 1000
            dic["solution"] = solution_loss
        return LossDict(dic)

    def infer(self, problem: ProblemInterface) -> np.ndarray:
        mesh_np = problem.get_mesh()
        mesh = torch.from_numpy(mesh_np).float().unsqueeze(dim=0)
        mesh = mesh.unsqueeze(0).to(self.device)

        descriptions_np = np.stack(problem.get_descriptions())
        description = torch.from_numpy(descriptions_np).float()
        description = description.unsqueeze(0).to(self.device)

        out, _ = self.forward((mesh, description))
        out_np = out.cpu().detach().numpy().flatten()
        return out_np


@dataclass
class VoxelAutoEncoderConfig(ModelConfigBase):
    dim_bottleneck: int = 1024


class VoxelAutoEncoder(ModelBase[VoxelAutoEncoderConfig]):
    encoder: nn.Sequential
    decoder: nn.Sequential

    class Reshape(nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.shape = args

        def forward(self, x):
            return x.view(self.shape)

    def _setup_from_config(self, config: VoxelAutoEncoderConfig) -> None:
        n_channel = 1
        # NOTE: DO NOT add bath normalization layer
        encoder_layers = [
            nn.Conv3d(n_channel, 8, (3, 3, 2), padding=1, stride=(2, 2, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, (3, 3, 3), padding=1, stride=(2, 2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, (3, 3, 3), padding=1, stride=(2, 2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, (3, 3, 3), padding=1, stride=(2, 2, 2)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4096, config.dim_bottleneck),
            nn.ReLU(inplace=True),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        # NOTE: DO NOT add bath normalization layer
        decoder_layers = [
            nn.Linear(config.dim_bottleneck, 4096),
            nn.ReLU(inplace=True),
            self.Reshape(-1, 64, 4, 4, 4),
            nn.ConvTranspose3d(64, 32, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, 4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 8, 4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(8, 1, (4, 4, 3), padding=1, stride=(2, 2, 1)),
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    def loss(self, sample: Tensor) -> LossDict:
        mesh = sample[0]
        encoded = self.encoder(mesh)
        reconst = self.decoder(encoded)
        loss = nn.MSELoss()(mesh, reconst)
        return LossDict({"reconstruction": loss})
