from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from mohou.model.common import LossDict, ModelBase, ModelConfigBase
from torch import Tensor


@dataclass
class IterationPredictorConfig(ModelConfigBase):
    dim_problem_descriptor: int
    dim_conv: int = 3

    def __post_init__(self):
        assert self.dim_conv in [2, 3]


class IterationPredictor(ModelBase[IterationPredictorConfig]):
    convnet: nn.Sequential
    linears: nn.Sequential
    margin: Optional[float] = None

    def _setup_from_config(self, config: IterationPredictorConfig) -> None:
        n_channel = 1
        n_conv_out_dim = 1000
        if config.dim_conv == 2:
            # use simple cnn
            encoder_layers = [
                nn.Conv2d(n_channel, 8, 3, padding=1, stride=(2, 2)),  # 14x14
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),
                nn.BatchNorm2d(64),
                nn.Flatten(),
                nn.Linear(1024, n_conv_out_dim),
                nn.ReLU(inplace=True),
            ]
            self.convnet = nn.Sequential(*encoder_layers)
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
                nn.Flatten(),
                nn.Linear(4096, n_conv_out_dim),
                nn.ReLU(inplace=True),
            ]
            self.convnet = nn.Sequential(*encoder_layers)

        n_input = n_conv_out_dim + config.dim_problem_descriptor

        self.linears = nn.Sequential(
            nn.Linear(n_input, n_conv_out_dim),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, sample: Tuple[Tensor, Tensor]) -> Tensor:
        mesh, descriptor = sample
        # descriptor: (n_batch x n_pose x dim_descriptor)
        n_batch, n_pose, _ = descriptor.shape
        if self.config.dim_conv == 3:
            # mesh: n_batch x 1 x (3d-size)
            assert mesh.dim() == 5
        elif self.config.dim_conv == 2:
            assert mesh.dim() == 4

        # mesh_features: (n_batch x 1 x dim_feature)
        mesh_features: torch.Tensor = self.convnet(mesh)
        # mesh_features_rep: (n_batch * n_pose x dim_feature)
        mesh_features_rep = mesh_features.repeat(n_pose, 1)

        # descriptor_flatten: (n_batch * n_pose x dim_descriptor)
        descriptor_flatten = descriptor.reshape(n_batch * n_pose, -1)

        vectors = torch.concat([mesh_features_rep, descriptor_flatten], dim=1)
        # out: (b_batch * n_pose)
        out = self.linears(vectors)
        out_original = out.reshape(n_batch, n_pose)
        return out_original

    def loss(self, sample: Tuple[Tensor, Tensor, Tensor]):
        mesh, descriptor, value = sample
        # mesh: (n_batch, (mesh_size))
        # descriptors: (n_batch, n_pose, (descriptor_dim))
        # values: (n_batch, n_pose, (,))

        pred = self.forward((mesh, descriptor))
        loss = nn.MSELoss()(pred, value)
        return LossDict({"prediction": loss})
