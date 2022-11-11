from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from mohou.model.common import LossDict, ModelBase, ModelConfigBase
from torch import Tensor

from hifuku.types import ProblemInterface


@dataclass
class IterationPredictorConfig(ModelConfigBase):
    dim_problem_descriptor: int
    dim_solution: int
    dim_conv: int = 3
    dim_description_expand: int = 300
    use_solution_pred: bool = False
    use_reconstruction: bool = True

    def __post_init__(self):
        assert self.dim_conv in [2, 3]


class IterationPredictor(ModelBase[IterationPredictorConfig]):
    encoder: nn.Sequential
    post_encoder: nn.Sequential
    decoder: Optional[nn.Sequential]
    linears: nn.Sequential
    iter_linears: nn.Sequential
    solution_linears: Optional[nn.Sequential]
    description_expand_lineras: nn.Sequential
    margin: Optional[float] = None

    def _setup_from_config(self, config: IterationPredictorConfig) -> None:
        n_channel = 1
        n_conv_out_dim = 1000
        if config.dim_conv == 2:
            assert False, "under construction"
            # use simple cnn
            # encoder_layers = [
            #    nn.Conv2d(n_channel, 8, 3, padding=1, stride=(2, 2)),  # 14x14
            #    nn.BatchNorm2d(8),
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),
            #    nn.BatchNorm2d(16),
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),
            #    nn.BatchNorm2d(32),
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),
            #    nn.BatchNorm2d(64),
            #    nn.Flatten(),
            #    nn.Linear(1024, n_conv_out_dim),
            #    nn.ReLU(inplace=True),
            # ]
            # self.encoder = nn.Sequential(*encoder_layers)

            # if config.use_reconstruction:
            #    assert False
            # else:
            #    self.decoder = None
        else:
            encoder_layers = [
                nn.Conv3d(n_channel, 8, (3, 3, 2), padding=1, stride=(2, 2, 1)),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 16, (3, 3, 3), padding=1, stride=(2, 2, 2)),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 32, (3, 3, 3), padding=1, stride=(2, 2, 2)),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 64, (3, 3, 3), padding=1, stride=(2, 2, 2)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
            ]
            self.encoder = nn.Sequential(*encoder_layers)

            post_encoder_layers = [
                nn.Flatten(),
                nn.Linear(4096, n_conv_out_dim),
                nn.ReLU(inplace=True),
            ]
            self.post_encoder = nn.Sequential(*post_encoder_layers)

            if config.use_reconstruction:
                decoder_layers = [
                    nn.ConvTranspose3d(64, 32, 3, padding=1, stride=2),
                    nn.BatchNorm3d(32),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(32, 16, 4, padding=1, stride=2),
                    nn.BatchNorm3d(16),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(16, 8, 4, padding=1, stride=2),
                    nn.BatchNorm3d(8),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(8, 1, (4, 4, 3), padding=1, stride=(2, 2, 1)),
                ]
                self.decoder = nn.Sequential(*decoder_layers)
            else:
                self.decoder = None

        self.description_expand_lineras = nn.Sequential(
            nn.Linear(config.dim_problem_descriptor, config.dim_description_expand),
            nn.ReLU(),
            nn.Linear(config.dim_description_expand, config.dim_description_expand),
            nn.ReLU(),
        )

        n_input = n_conv_out_dim + config.dim_description_expand

        self.linears = nn.Sequential(
            nn.Linear(n_input, n_conv_out_dim),
            nn.ReLU(),
            nn.Linear(1000, 500),
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

    def forward(
        self, sample: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        mesh, descriptor = sample
        # descriptor: (n_batch x n_pose x dim_descriptor)
        n_batch, n_pose, _ = descriptor.shape
        if self.config.dim_conv == 3:
            # mesh: n_batch x 1 x (3d-size)
            assert mesh.dim() == 5
        elif self.config.dim_conv == 2:
            assert mesh.dim() == 4

        encoded = self.encoder(mesh)

        # mesh_features: (n_batch x 1 x dim_feature)
        mesh_features: torch.Tensor = self.post_encoder(encoded)
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

        if self.decoder is not None:
            mesh_decoded = self.decoder(encoded)
        else:
            mesh_decoded = None

        if self.solution_linears is not None:
            # solution_pred: (n_batch * n_pose x dim_solution)
            solution_pred = self.solution_linears(tmp)
            # solution_pred: (n_batch x n_pose x dim_solution)
            solution_pred = solution_pred.reshape(n_batch, n_pose, -1)
        else:
            solution_pred = None
        return iter_pred, mesh_decoded, solution_pred

    def loss(self, sample: Tuple[Tensor, Tensor, Tensor, Tensor]) -> LossDict:
        mesh, descriptor, iterval, solution = sample
        # mesh: (n_batch, (mesh_size))
        # descriptors: (n_batch, n_pose, (descriptor_dim))
        # iterval: (n_batch, n_pose, (,))

        dic = {}
        iter_pred, mesh_decoded, solution_pred = self.forward((mesh, descriptor))
        iter_loss = nn.MSELoss()(iter_pred, iterval)
        dic["iter"] = iter_loss

        if mesh_decoded is not None:
            reconstruction_loss = nn.MSELoss()(mesh, mesh_decoded)
            dic["reconstruction"] = reconstruction_loss

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

        out, _, _ = self.forward((mesh, description))
        out_np = out.cpu().detach().numpy().flatten()
        return out_np
