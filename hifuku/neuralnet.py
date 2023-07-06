from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import tqdm
from mohou.model.common import LossDict, ModelBase, ModelConfigBase
from mohou.utils import detect_device
from rpbench.interface import DescriptionTable
from skmp.trajectory import Trajectory
from torch import Tensor
from torch.utils.data import Dataset, default_collate

from hifuku.llazy.dataset import LazyDecomplessDataLoader, LazyDecomplessDataset
from hifuku.types import RawData


class AutoEncoderBase(ABC):
    @abstractmethod
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def n_bottleneck(self) -> int:
        ...

    @property
    @abstractmethod
    def trained(self) -> bool:
        ...

    @abstractmethod
    def get_device(self) -> torch.device:
        ...

    @abstractmethod
    def put_on_device(self, device: torch.device) -> None:
        ...


class NullAutoEncoder(AutoEncoderBase):
    device: torch.device

    def __init__(self):
        self.device = torch.device("cpu")

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        # assume that X is empty tensor
        assert X.ndim == 2
        n_batch, dummy = X.shape
        assert dummy == 0
        return torch.empty((n_batch, 0)).to(self.device)

    @property
    def n_bottleneck(self) -> int:
        return 0

    @property
    def trained(self) -> bool:
        return True

    def put_on_device(self, device: torch.device) -> None:
        self.device = device

    def get_device(self) -> torch.device:
        return self.device


@dataclass
class IterationPredictorDataset(Dataset):
    mesh_encodeds: Optional[torch.Tensor]
    descriptions: torch.Tensor
    itervals: torch.Tensor
    _problem_per_sample: int

    # TODO: __add__
    def add(self, other: "IterationPredictorDataset") -> None:
        if self.mesh_encodeds is not None:
            assert other.mesh_encodeds is not None
            mesh_encodeds = torch.vstack([self.mesh_encodeds, other.mesh_encodeds])
        else:
            assert other.mesh_encodeds is None
            mesh_encodeds = None
        descriptions = torch.vstack([self.descriptions, other.descriptions])
        itervals = torch.hstack([self.itervals, other.itervals])

        self.mesh_encodeds = mesh_encodeds
        self.descriptions = descriptions
        self.itervals = itervals

    def __len__(self) -> int:
        return len(self.descriptions)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        if self.mesh_encodeds is None:
            mesh_encoded_here = torch.empty(0)
        else:
            mesh_encoded_here = self.mesh_encodeds[idx // self._problem_per_sample]
        # Note: mesh_encoded is (n_sample, size) shape, though
        # otheres have shape of (n_sample * n_problem, size).
        # Thus we must devide the dix by n_problem
        return (
            mesh_encoded_here,
            self.descriptions[idx],
            self.itervals[idx],
        )

    @classmethod
    def load_from_path(
        cls, dataset_path: Path, ae_model: AutoEncoderBase
    ) -> "IterationPredictorDataset":
        dataset = LazyDecomplessDataset.load(dataset_path, RawData, n_worker=-1)
        loader = LazyDecomplessDataLoader(dataset, batch_size=1000, shuffle=False)
        return cls.construct(loader, ae_model)

    @classmethod
    def construct_from_tasks_and_resultss(
        cls, init_solution, tasks, resultss, solver_config, ae_model: AutoEncoderBase
    ) -> "IterationPredictorDataset":
        raw_data_list = []
        for task, results in zip(tasks, resultss):
            raw_data = RawData(init_solution, task.export_table(), results, solver_config)
            raw_data_list.append(raw_data)
        zipped = [raw_data.to_tensors() for raw_data in raw_data_list]
        sample = default_collate(zipped)
        return cls.construct([sample], ae_model)

    @classmethod
    def construct(cls, loader, ae_model: AutoEncoderBase) -> "IterationPredictorDataset":
        device = detect_device()
        ae_model.put_on_device(device)

        encoded_list = []
        description_list = []
        iterval_list = []

        # create minibatch list
        n_problem: int = 0
        mesh_used: bool = False  # dirty. set in the for loop

        for sample in tqdm.tqdm(loader):
            mesh, description, iterval = sample

            mesh_used = mesh is not None

            if mesh_used:
                mesh = mesh.to(device)  # n_batch x (*shape)
                encoded: torch.Tensor = ae_model.encode(mesh).detach().cpu()

            n_batch, n_problem, _ = description.shape

            for i in range(n_batch):

                if mesh_used:
                    encoded_list.append(encoded[i].unsqueeze(dim=0))

                description_list.append(description[i])
                iterval_list.append(iterval[i])
        assert n_problem > 0

        if mesh_used:
            mesh_encodeds_concat = torch.cat(encoded_list, dim=0)  # n_batch x n_bottleneck
        else:
            mesh_encodeds_concat = None

        descriptions_concat = torch.cat(description_list, dim=0)
        itervals_concat = torch.cat(iterval_list, dim=0)

        n_data = len(descriptions_concat)
        assert len(itervals_concat) == n_data
        return cls(mesh_encodeds_concat, descriptions_concat, itervals_concat, n_problem)


@dataclass
class IterationPredictorConfig(ModelConfigBase):
    dim_problem_descriptor: int
    dim_conv_bottleneck: int
    n_layer1_width: int = 500
    n_layer2_width: int = 100
    n_layer3_width: int = 50
    dim_description_expand: Optional[int] = 300
    use_solution_pred: bool = False


class IterationPredictor(ModelBase[IterationPredictorConfig]):
    linears: nn.Sequential
    description_expand_linears: Optional[nn.Sequential]
    initial_solution: Optional[Trajectory] = None

    def _setup_from_config(self, config: IterationPredictorConfig) -> None:

        if config.dim_description_expand is not None:
            self.description_expand_linears = nn.Sequential(
                nn.Linear(config.dim_problem_descriptor, config.dim_description_expand),
                nn.ReLU(),
                nn.Linear(config.dim_description_expand, config.dim_description_expand),
                nn.ReLU(),
            )
            n_input = config.dim_conv_bottleneck + config.dim_description_expand
        else:
            self.description_expand_linears = None
            n_input = config.dim_conv_bottleneck + config.dim_problem_descriptor

        self.linears = nn.Sequential(
            nn.Linear(n_input, config.n_layer1_width),
            nn.ReLU(),
            nn.Linear(config.n_layer1_width, config.n_layer2_width),
            nn.ReLU(),
            nn.Linear(config.n_layer2_width, config.n_layer3_width),
            nn.ReLU(),
            nn.Linear(config.n_layer3_width, 1),
        )

    def forward(self, sample: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        mesh_features, descriptor = sample
        if self.description_expand_linears is not None:
            descriptor = self.description_expand_linears(descriptor)

        vectors = torch.concat([mesh_features, descriptor], dim=1)
        iter_pred = self.linears(vectors)
        solution_pred = None
        return iter_pred, solution_pred

    def loss(self, sample: Tuple[Tensor, Tensor, Tensor]) -> LossDict:
        mesh_encoded, descriptor, iterval = sample
        # mesh: (n_batch, (mesh_size))
        # descriptors: (n_batch, n_pose, (descriptor_dim))
        # iterval: (n_batch, n_pose, (,))
        dic = {}
        iter_pred, solution_pred = self.forward((mesh_encoded, descriptor))
        if iterval.ndim == 1:
            iterval = iterval.unsqueeze(dim=1)
        iter_loss = nn.MSELoss()(iter_pred, iterval)
        dic["iter"] = iter_loss

        if solution_pred is not None:
            assert False, "this feature is deleted"
            solution = solution_pred  # this is just a dummy line to linter pass
            solution_loss = nn.MSELoss()(solution_pred, solution) * 1000
            dic["solution"] = solution_loss
        return LossDict(dic)


@dataclass
class AutoEncoderConfig(ModelConfigBase):
    dim_bottleneck: int = 1024


VoxelAutoEncoderConfig = AutoEncoderConfig  # for backword compatibility TODO: remove this


class NeuralAutoEncoderBase(ModelBase[AutoEncoderConfig], AutoEncoderBase):
    encoder: nn.Sequential
    decoder: nn.Sequential
    loss_called: bool = False  # flag to show the model is trained

    class Reshape(nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.shape = args

        def forward(self, x):
            return x.view(self.shape)

    @property
    def trained(self) -> bool:
        return self.loss_called

    def get_device(self) -> torch.device:
        return self.device

    def encode(self, mesh: Tensor) -> Tensor:
        return self.encoder(mesh)

    def loss(self, mesh: Tensor) -> LossDict:
        self.loss_called = True
        encoded = self.encoder(mesh)
        reconst = self.decoder(encoded)
        loss = nn.MSELoss()(mesh, reconst)
        return LossDict({"reconstruction": loss})

    @property
    def n_bottleneck(self) -> int:
        return self.config.dim_bottleneck

    @abstractmethod
    def _setup_from_config(self, config: AutoEncoderConfig) -> None:
        ...


class VoxelAutoEncoder(NeuralAutoEncoderBase):
    def _setup_from_config(self, config: AutoEncoderConfig) -> None:
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


class PixelAutoEncoder(NeuralAutoEncoderBase):
    def _setup_from_config(self, config: AutoEncoderConfig) -> None:
        # suppose n_pixel = 112
        n_channel = 1
        encoder_layers = [
            nn.Conv2d(n_channel, 8, 3, padding=1, stride=(2, 2)),  # 56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),  # 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),  # 7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=(2, 2)),  # 4x4
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 16, config.dim_bottleneck),
            nn.ReLU(inplace=True),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Linear(config.dim_bottleneck, 128 * 16),
            nn.ReLU(inplace=True),
            self.Reshape(-1, 128, 4, 4),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, n_channel, 4, stride=2, padding=1),
        ]
        self.decoder = nn.Sequential(*decoder_layers)
