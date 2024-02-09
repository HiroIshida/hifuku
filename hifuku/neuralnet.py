import logging
import os
import pickle
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import tqdm
from mohou.model.common import LossDict, ModelBase, ModelConfigBase
from mohou.utils import detect_device
from skmp.trajectory import Trajectory
from torch import Tensor
from torch.utils.data import Dataset, default_collate

from hifuku.types import RawData
from hifuku.utils import determine_process_thread

logger = logging.getLogger(__name__)


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
    mesh_likes: Optional[torch.Tensor]
    descriptions: torch.Tensor
    itervals: torch.Tensor
    bools_fail: torch.Tensor
    weights: torch.Tensor
    n_inner: int
    encoded: bool
    indices_remain: Optional[torch.Tensor] = None
    """
    mesh_likes can be either a of stack of feature vectors (n_elem, n_feature)
    or stack of meshes (n_elem, ...) depending on `encoded".
    """

    def reduce(self, target_ratio: float = 0.7) -> None:
        indices_fails = torch.where(self.bools_fail)[0]
        false_rate_now = len(indices_fails) / len(self)
        logger.info(f"current false rate: {false_rate_now}")
        if false_rate_now < target_ratio:
            return

        n_success_now = len(self) - len(indices_fails)
        target_n_false = int(target_ratio * n_success_now / (1 - target_ratio))
        n_false_reduce = len(indices_fails) - target_n_false

        # randomize indices_fail
        indices_fails = indices_fails[torch.randperm(len(indices_fails))]
        indices_remove = indices_fails[:n_false_reduce]

        # show first 100 indices removed
        logger.info(f"indices removed: {indices_remove[:100]}...{indices_remove[-1]}")

        bools_remain = torch.ones(len(self), dtype=torch.bool)
        bools_remain[indices_remove] = False
        indices_remain = torch.where(bools_remain)[0]
        self.indices_remain = indices_remain

        # compute current false rate
        indices_fails = torch.where(self.bools_fail[indices_remain])[0]
        false_rate_now = len(indices_fails) / len(indices_remain)
        logger.info(f"reduced to {torch.sum(self.bools_fail)} sample from {len(self.bools_fail)}")
        logger.info(f"false rate: {false_rate_now}")

    # TODO: __add__
    def add(self, other: "IterationPredictorDataset") -> None:
        if self.mesh_likes is not None:
            assert other.mesh_likes is not None
            mesh_likes = torch.vstack([self.mesh_likes, other.mesh_likes])
        else:
            assert other.mesh_likes is None
            mesh_likes = None
        descriptions = torch.vstack([self.descriptions, other.descriptions])
        itervals = torch.hstack([self.itervals, other.itervals])

        self.mesh_likes = mesh_likes
        self.descriptions = descriptions
        self.itervals = itervals

    @property
    def n_task(self) -> int:
        return int(len(self) / self.n_inner)

    def __len__(self) -> int:
        if self.indices_remain is not None:
            return len(self.indices_remain)
        return len(self.descriptions)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        if self.indices_remain is not None:
            idx = self.indices_remain[idx]

        if self.mesh_likes is None:
            mesh_like_here = torch.empty(0)
        else:
            # Note: len(mesh_likes) * n_inner = len(descriptions)
            # because n_inner descriptions share a single mesh.
            # Thus we must devide the dix by n_inner
            mesh_like_here = self.mesh_likes[idx // self.n_inner]
        return (
            mesh_like_here,
            self.descriptions[idx],
            self.itervals[idx],
            self.weights[idx],
        )

    @staticmethod
    def _initialize(init_solution, solver_config):
        global _init_solution_, _solver_config_
        _init_solution_ = init_solution  # type: ignore
        _solver_config_ = solver_config  # type: ignore
        logger.info(f"initialized pid {os.getpid()}")

    @staticmethod
    def _create_raw_data(pair):  # used in ProcessPool
        task, results = pair
        global _init_solution_, _solver_config_
        raw_data = RawData(_init_solution_, task.export_table(), results, _solver_config_)  # type: ignore
        return raw_data

    @classmethod
    def _compute_expected_raw_data_size(cls, tasks, resultss, solver_config) -> int:
        task = tasks[0]
        results = resultss[0]
        raw_data = RawData(None, task.export_table(), results, solver_config)  # type: ignore
        dump_path = Path(f"/tmp/expected_raw_data_{uuid.uuid4()}.pkl")
        logger.debug(f"dumping raw data to {dump_path}")
        with dump_path.open("wb") as f:
            pickle.dump(raw_data, f)
        expected_total_size = len(pickle.dumps(raw_data)) * len(tasks)  # in bytes
        return expected_total_size

    @classmethod
    def construct_from_tasks_and_resultss(
        cls,
        init_solution,
        tasks,
        resultss,
        solver_config,
        weightss: Optional[Tensor],
        ae_model: Optional[AutoEncoderBase],
    ) -> "IterationPredictorDataset":
        expected_total_data_size = cls._compute_expected_raw_data_size(
            tasks, resultss, solver_config
        )
        logger.info(f"expected total data size: {expected_total_data_size / 1e6} MB")

        raw_data_list = []
        n_process, _ = determine_process_thread()
        with ProcessPoolExecutor(
            n_process, initializer=cls._initialize, initargs=(init_solution, solver_config)
        ) as executor:
            args = list(zip(tasks, resultss))
            for raw_data in tqdm.tqdm(executor.map(cls._create_raw_data, args), total=len(tasks)):
                raw_data_list.append(raw_data)

        zipped = [raw_data.to_tensors() for raw_data in raw_data_list]
        sample = default_collate(zipped)
        if weightss is None:
            weightss = torch.ones((len(tasks), tasks[0].n_inner_task))
        assert weightss.ndim == 2
        return cls.construct([sample], weightss, ae_model)

    @classmethod
    def construct(
        cls, loader_like, weightss: Tensor, ae_model: Optional[AutoEncoderBase]
    ) -> "IterationPredictorDataset":
        if ae_model is None:
            return cls.construct_keeping_mesh(loader_like, weightss)
        else:
            return cls.construct_by_encoding(loader_like, weightss, ae_model)

    @classmethod
    def construct_keeping_mesh(cls, loader_like, weightss: Tensor) -> "IterationPredictorDataset":
        meshes_list = []
        descriptions_stacked_list = []
        iterval_stacked_list = []
        bools_fail_stacked_list = []

        n_inner = None
        for sample in tqdm.tqdm(loader_like):
            meshes, descriptionss, itervals, bools_fail = sample
            meshes_list.append(meshes)
            descriptions_stacked_list.append(descriptionss.reshape((-1, descriptionss.shape[-1])))
            iterval_stacked_list.append(itervals.flatten())
            bools_fail_stacked_list.append(bools_fail.flatten())

            n_inner = len(descriptionss[0])
        assert n_inner is not None

        return cls(
            torch.concat(meshes_list),
            torch.concat(descriptions_stacked_list),
            torch.hstack(iterval_stacked_list),
            torch.hstack(bools_fail_stacked_list),
            weightss.flatten(),
            n_inner,
            False,
        )

    @classmethod
    def construct_by_encoding(
        cls, loader_like, weightss: Tensor, ae_model: AutoEncoderBase
    ) -> "IterationPredictorDataset":
        device = detect_device()
        ae_model.put_on_device(device)

        encoded_list = []
        description_list = []
        iterval_list = []
        bools_fail_list = []

        # create minibatch list
        n_problem: int = 0
        mesh_used: bool = False  # dirty. set in the for loop

        for sample in tqdm.tqdm(loader_like):
            mesh, description, iterval, bools_fail = sample

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
                bools_fail_list.append(bools_fail[i])
        assert n_problem > 0

        if mesh_used:
            mesh_encodeds_concat = torch.cat(encoded_list, dim=0)  # n_batch x n_bottleneck
        else:
            mesh_encodeds_concat = None

        descriptions_concat = torch.cat(description_list, dim=0)
        itervals_concat = torch.cat(iterval_list, dim=0)
        bools_fails_concat = torch.cat(bools_fail_list, dim=0)

        n_data = len(descriptions_concat)
        assert len(itervals_concat) == n_data
        return cls(
            mesh_encodeds_concat,
            descriptions_concat,
            itervals_concat,
            bools_fails_concat,
            weightss.flatten(),
            n_problem,
            True,
        )


@dataclass
class IterationPredictorConfig(ModelConfigBase):
    dim_problem_descriptor: int
    dim_conv_bottleneck: int
    n_layer1_width: int = 500
    n_layer2_width: int = 100
    n_layer3_width: int = 50
    layers: Optional[List[int]] = None
    dim_description_expand: Optional[int] = 50
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

        if config.layers is None:
            self.linears = nn.Sequential(
                nn.Linear(n_input, config.n_layer1_width),
                nn.ReLU(),
                nn.Linear(config.n_layer1_width, config.n_layer2_width),
                nn.ReLU(),
                nn.Linear(config.n_layer2_width, config.n_layer3_width),
                nn.ReLU(),
                nn.Linear(config.n_layer3_width, 1),
            )
        else:
            width_list = [n_input] + config.layers
            layers: List[Any] = []
            for i in range(len(width_list) - 1):
                layers.append(nn.Linear(width_list[i], width_list[i + 1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(config.layers[-1], 1))
            self.linears = nn.Sequential(*layers)

    def forward(self, sample: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        mesh_features, descriptor = sample
        if self.description_expand_linears is not None:
            descriptor = self.description_expand_linears(descriptor)

        vectors = torch.concat([mesh_features, descriptor], dim=1)
        iter_pred = self.linears(vectors)
        solution_pred = None
        return iter_pred, solution_pred

    def loss(self, sample: Tuple[Tensor, Tensor, Tensor, Tensor]) -> LossDict:
        mesh_encoded, descriptor, iterval, weight = sample
        iter_pred, _ = self.forward((mesh_encoded, descriptor))
        iter_pred = iter_pred.flatten()
        iterval = iterval.flatten()

        weihted_iter_loss = (iter_pred - iterval) ** 2 * weight
        iter_loss = torch.mean(weihted_iter_loss)
        dic = {"iter": iter_loss}
        return LossDict(dic)


@dataclass
class AutoEncoderConfig(ModelConfigBase):
    dim_bottleneck: int = 200
    n_grid: Literal[56, 112] = 112


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
        if config.n_grid == 112:
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
        elif config.n_grid == 56:
            n_channel = 1
            encoder_layers = [
                nn.Conv2d(n_channel, 8, 3, padding=1, stride=(2, 2)),  # 14x14
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(1024, config.dim_bottleneck),
                nn.ReLU(inplace=True),
            ]

            decoder_layers = [
                nn.Linear(config.dim_bottleneck, 1024),
                nn.ReLU(inplace=True),
                self.Reshape(-1, 64, 4, 4),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 4, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(4, 1, 4, stride=2, padding=1),
            ]
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)


@dataclass
class IterationPredictorWithEncoderConfig(ModelConfigBase):
    # A very bad design. but I don't have time. hahhahaha...
    iterpred_model: IterationPredictor
    ae_model: AutoEncoderBase

    def __post_init__(self):
        assert not isinstance(self.ae_model, NullAutoEncoder)


class IterationPredictorWithEncoder(ModelBase[IterationPredictorWithEncoderConfig]):
    iterpred_model: IterationPredictor
    ae_model: AutoEncoderBase

    def put_on_device(self, device: Optional[torch.device] = None):
        super().put_on_device(device)
        self.iterpred_model.put_on_device(device)

        # NOTE: ae_model cannot be nullautoencoder thus we put type-ignore
        self.ae_model.put_on_device(device)  # type: ignore

    def _setup_from_config(self, config: IterationPredictorWithEncoderConfig) -> None:
        self.iterpred_model = config.iterpred_model
        self.ae_model = config.ae_model

    def forward_multi_inner(self, mesh: Tensor, descs: Tensor) -> Tensor:
        # descs shares the same mesh
        n_inner, _ = descs.shape
        encoded = self.ae_model.encode(mesh)
        encoded_repeated = encoded.repeat((n_inner, 1))
        iters_pred, _ = self.iterpred_model.forward((encoded_repeated, descs))
        return iters_pred

    def forward(self, sample: Tuple[Tensor, Tensor]) -> Tensor:
        meshes, descs = sample
        encoded = self.ae_model.encode(meshes)
        iters_pred, _ = self.iterpred_model.forward((encoded, descs))
        iters_pred = iters_pred.flatten()
        return iters_pred

    def loss(self, sample: Tuple[Tensor, Tensor, Tensor, Tensor]) -> LossDict:
        meshes, descs, iters, weithtss = sample
        iters_pred = self.forward((meshes, descs))
        iters_pred = iters_pred.flatten()
        iters = iters.flatten()

        weihted_iter_loss = (iters_pred - iters) ** 2 * weithtss.flatten()
        iter_loss = torch.mean(weihted_iter_loss)
        dic = {"iter": iter_loss}
        return LossDict(dic)
