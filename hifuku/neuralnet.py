import copy
import logging
import multiprocessing
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import threadpoolctl
import torch
import torch.nn as nn
import tqdm
from mohou.model.common import LossDict, ModelBase, ModelConfigBase
from rpbench.interface import TaskBase
from skmp.solver.interface import ConfigProtocol, ResultProtocol
from skmp.trajectory import Trajectory
from torch import Tensor
from torch.utils.data import Dataset

from hifuku.utils import num_torch_thread

logger = logging.getLogger(__name__)


class AutoEncoderBase(ABC):
    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
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


CompressedBytes = Tuple[bytes, Tuple[int, ...]]  # tuple for shape as we need it to reconstruct


@dataclass
class CostPredictorDataset(Dataset):
    mesh_likes: Union[List[CompressedBytes], torch.Tensor, None]
    descriptions: torch.Tensor
    costs: torch.Tensor
    weights: torch.Tensor
    """
    mesh_likes can be either a of stack of feature vectors (n_elem, n_feature)
    or stack of meshes (n_elem, ...) depending on `encoded".
    """

    # TODO: __add__
    def add(self, other: "CostPredictorDataset") -> None:
        import torch  # I don't know why but this is necessary if used in dill

        if self.mesh_likes is not None:
            assert other.mesh_likes is not None
            if isinstance(self.mesh_likes, list):
                assert isinstance(other.mesh_likes, list)
                mesh_likes = self.mesh_likes + other.mesh_likes
            else:
                mesh_likes = torch.vstack([self.mesh_likes, other.mesh_likes])
        else:
            assert other.mesh_likes is None
            mesh_likes = None
        descriptions = torch.vstack([self.descriptions, other.descriptions])
        costs = torch.hstack([self.costs, other.costs])
        weights = torch.hstack([self.weights, other.weights])

        self.mesh_likes = mesh_likes
        self.descriptions = descriptions
        self.costs = costs
        self.weights = weights

    def __len__(self) -> int:
        return len(self.descriptions)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        if self.mesh_likes is None:
            mesh_like_here = torch.empty(0)
        else:
            mesh_like_here = self.mesh_likes[idx]
            if isinstance(mesh_like_here, tuple):  # meaning that it's compressed
                b, shape = mesh_like_here
                mesh_like_here = torch.from_numpy(
                    np.frombuffer(zlib.decompress(b), dtype=np.float32)
                )
                full_shape = (1,) + shape
                mesh_like_here = mesh_like_here.view(full_shape)

        return (
            mesh_like_here,
            self.descriptions[idx],
            self.costs[idx],
            self.weights[idx],
        )


def create_dataset_from_params_and_results(
    task_params: np.ndarray,
    results: List[ResultProtocol],
    solver_config,
    task_type: Type[TaskBase],
    weights: Optional[Tensor],
    ae_model: Optional[AutoEncoderBase],
    clamp_factor: float = 2.0,
    compress_mesh: bool = False,
    n_max: int = 1000000,
) -> CostPredictorDataset:
    n_param = len(task_params)
    n_split = n_param // n_max + 1
    indices_list = np.array_split(np.arange(n_param), n_split)
    dataset_all = None
    for indices in indices_list:
        dataset = create_dataset_from_params_and_results_sub(
            task_params[indices],
            np.array(results)[indices],
            solver_config,
            task_type,
            weights[indices] if weights is not None else None,
            ae_model,
            clamp_factor,
            compress_mesh,
        )
        if dataset_all is None:
            dataset_all = dataset
        else:
            dataset_all.add(dataset)
            print(f"adding to dataset {len(dataset_all)}")
    return dataset_all


def create_dataset_from_params_and_results_sub(
    task_params: np.ndarray,
    results: List[ResultProtocol],
    solver_config,
    task_type: Type[TaskBase],
    weights: Optional[Tensor],
    ae_model: Optional[AutoEncoderBase],
    clamp_factor: float = 2.0,
    compress_mesh: bool = False,
) -> CostPredictorDataset:

    if ae_model is not None:  # meaning image is encoded to vector
        if compress_mesh:
            raise ValueError("it's meaningless to compress mesh if image is encoded")

    n_process = 6
    # use multiprocessing.
    # split the data for each process
    n_data = len(task_params)

    split_indices_list = np.array_split(np.arange(n_data), n_process)
    split_indices_list = [indices for indices in split_indices_list if len(indices) > 0]
    task_params_list = [task_params[indices] for indices in split_indices_list]
    results_list = [list(np.array(results)[indices]) for indices in split_indices_list]
    if weights is None:
        weights_list = [None] * n_process
    else:
        weights_list = [weights[indices] for indices in split_indices_list]

    if ae_model is not None:
        ae_model_copied = copy.deepcopy(ae_model)
        ae_model_copied.put_on_device(torch.device("cpu"))
    else:
        ae_model_copied = None

    # spawn is maybe necessary to avoid the error in torch multiprocessing
    with multiprocessing.get_context("spawn").Pool(n_process) as pool:
        dataset_list = pool.starmap(
            _create_dataset_from_params_and_results,
            [
                (
                    task_params,
                    results,
                    solver_config,
                    task_type,
                    weights,
                    ae_model_copied,
                    clamp_factor,
                    compress_mesh,
                )
                for task_params, results, weights in zip(
                    task_params_list, results_list, weights_list
                )
            ],
        )

    dataset = dataset_list[0]
    for d in dataset_list[1:]:
        dataset.add(d)
    return dataset


def _create_dataset_from_params_and_results(
    task_params: np.ndarray,
    results,
    solver_config: ConfigProtocol,
    task_type: Type[TaskBase],
    weights: Optional[Tensor],
    ae_model: Optional[AutoEncoderBase],
    clamp_factor: float,
    compress_mesh: bool,
) -> CostPredictorDataset:
    def get_clamped_cost(result) -> int:
        if result.traj is None:
            return int(solver_config.n_max_call * clamp_factor)
        return result.n_call

    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        with num_torch_thread(1):
            if ae_model is not None:
                assert ae_model.get_device() == torch.device("cpu"), "to parallelize"

            if weights is None:
                weights = torch.ones(len(task_params))

            # preallocate memory for each data type
            # dummy run to determine the size of mesh_like_stacked
            task = task_type.from_task_param(task_params[0])
            expression = task.export_task_expression(use_matrix=True)
            matrix = expression.get_matrix()
            mesh_likes = None
            if matrix is not None:
                if ae_model is None:
                    if compress_mesh:
                        mesh_likes = [None] * len(task_params)
                    else:
                        mat_size = torch.from_numpy(matrix).float().unsqueeze(0).size()
                        stacked_size = (len(task_params),) + mat_size
                        mesh_likes = torch.zeros(stacked_size)
                else:
                    # Do not consider voxel tensor as matrix with channels
                    has_channel = len(matrix.shape) == 3 and not isinstance(
                        ae_model, VoxelAutoEncoder
                    )
                    mat_torch = torch.from_numpy(matrix).float()
                    if not has_channel:
                        mat_torch = mat_torch.unsqueeze(0)
                    encoded = ae_model.forward(mat_torch.unsqueeze(0)).squeeze(0).detach()
                    stacked_size = (len(task_params),) + encoded.size()
                    mesh_likes = torch.zeros(stacked_size)

            vector_size = expression.get_vector().size
            vector_stacked = torch.zeros((len(task_params), vector_size))
            costs = torch.zeros(len(task_params))

            for i in tqdm.tqdm(range(len(task_params))):
                task_param = task_params[i]
                result = results[i]

                task = task_type.from_task_param(task_param)
                expression = task.export_task_expression(use_matrix=True)
                matrix = expression.get_matrix()
                encoded: Union[torch.Tensor, CompressedBytes, None] = None
                if matrix is not None:
                    mat_torch = torch.from_numpy(matrix).float().unsqueeze(0)
                    if ae_model is None:
                        if compress_mesh:
                            assert isinstance(matrix, np.ndarray)
                            matrix_32 = matrix.astype(np.float32)
                            b = zlib.compress(matrix_32.tobytes())
                            encoded = (b, matrix.shape)
                        else:
                            encoded = mat_torch  # just don't encode
                    else:
                        # Do not consider voxel tensor as matrix with channels
                        has_channel = len(matrix.shape) == 3 and not isinstance(
                            ae_model, VoxelAutoEncoder
                        )
                        if not has_channel:
                            mat_torch = mat_torch.unsqueeze(0)
                        encoded = ae_model.forward(mat_torch).squeeze(0).detach()
                    assert mesh_likes is not None
                    mesh_likes[i] = encoded

                vector = expression.get_vector()
                vector_torch = torch.from_numpy(vector).float()
                vector_stacked[i] = vector_torch
                cost = get_clamped_cost(result)
                costs[i] = cost

            return CostPredictorDataset(mesh_likes, vector_stacked, costs, weights)


@dataclass
class CostPredictorConfig(ModelConfigBase):
    dim_task_descriptor: int
    dim_conv_bottleneck: int
    layers: Tuple[int, ...] = (500, 100, 100, 100, 50)
    dim_description_expand: Optional[int] = 50
    use_solution_pred: bool = False
    use_batch_norm: bool = True
    as_classifier: bool = False  # TODO: dirty design. we should create another class
    classifier_threshold: Optional[float] = None


class CostPredictor(ModelBase[CostPredictorConfig]):
    linears: nn.Sequential
    description_expand_linears: Optional[nn.Sequential]
    initial_solution: Optional[Trajectory] = None

    def _setup_from_config(self, config: CostPredictorConfig) -> None:
        if config.as_classifier:
            assert config.classifier_threshold is not None

        if config.dim_description_expand is not None:
            lst: List[nn.Module] = [
                nn.Linear(config.dim_task_descriptor, config.dim_description_expand)
            ]
            if config.use_batch_norm:
                lst.append(nn.BatchNorm1d(config.dim_description_expand))
            lst.append(nn.ReLU())
            lst.append(nn.Linear(config.dim_description_expand, config.dim_description_expand))
            if config.use_batch_norm:
                lst.append(nn.BatchNorm1d(config.dim_description_expand))
            lst.append(nn.ReLU())
            self.description_expand_linears = nn.Sequential(*lst)
            n_input = config.dim_conv_bottleneck + config.dim_description_expand
        else:
            self.description_expand_linears = None
            n_input = config.dim_conv_bottleneck + config.dim_task_descriptor

        width_list = [n_input] + list(config.layers)
        layers: List[Any] = []
        for i in range(len(width_list) - 1):
            layers.append(nn.Linear(width_list[i], width_list[i + 1]))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(width_list[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config.layers[-1], 1))
        if config.as_classifier:
            layers.append(nn.Sigmoid())
        self.linears = nn.Sequential(*layers)

    def forward(self, sample: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        mesh_features, descriptor = sample
        if self.description_expand_linears is not None:
            descriptor = self.description_expand_linears(descriptor)

        vectors = torch.concat([mesh_features, descriptor], dim=1)
        cost_pred = self.linears(vectors)

        solution_pred = torch.empty(0)  # dummy for backward compatibility
        return cost_pred, solution_pred

    def loss(self, sample: Tuple[Tensor, Tensor, Tensor, Tensor]) -> LossDict:
        mesh_encoded, descriptor, cost, weight = sample
        cost_pred, _ = self.forward((mesh_encoded, descriptor))
        cost_pred = cost_pred.flatten()
        if self.config.as_classifier:
            classes = (cost < self.config.classifier_threshold).float()
            class_loss = nn.BCELoss()(cost_pred, classes)
            dic = {"class": class_loss}
        else:
            cost = cost.flatten()

            weihted_cost_loss = (cost_pred - cost) ** 2 * weight
            cost_loss = torch.mean(weihted_cost_loss)
            dic = {"cost": cost_loss}
        return LossDict(dic)


@dataclass
class AutoEncoderConfig(ModelConfigBase):
    dim_bottleneck: int = 200
    n_grid: Literal[56, 112] = 112
    output_binary: bool = False
    n_channel: int = 1


VoxelAutoEncoderConfig = AutoEncoderConfig  # for backword compatibility TODO: remove this


class ReconstructionLossMixin:
    def loss(self, mesh: Tensor) -> LossDict:
        self.loss_called = True
        encoded = self.encoder(mesh)
        reconst = self.decoder(encoded)
        loss = nn.MSELoss()(mesh, reconst)
        return LossDict({"reconstruction": loss})


class ReferenceComparisonMixin:
    def loss(self, sample) -> LossDict:
        mesh, against = sample
        self.loss_called = True
        encoded = self.encoder(mesh)
        reconst = self.decoder(encoded)
        loss = nn.MSELoss()(against, reconst)
        return LossDict({"comparison": loss})


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class NeuralAutoEncoderBase(ModelBase[AutoEncoderConfig], AutoEncoderBase):
    encoder: nn.Sequential
    decoder: nn.Sequential
    loss_called: bool = False  # flag to show the model is trained

    class Reshape(nn.Module):
        # This class is no longer necessary (defined above)
        # but I keep it for pickle compatibility
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

    def forward(self, mesh: Tensor) -> Tensor:
        return self.encoder(mesh)

    @property
    def n_bottleneck(self) -> int:
        return self.config.dim_bottleneck

    @abstractmethod
    def _setup_from_config(self, config: AutoEncoderConfig) -> None:
        ...


class VoxelAutoEncoder(ReferenceComparisonMixin, NeuralAutoEncoderBase):
    def _setup_from_config(self, config: AutoEncoderConfig) -> None:
        assert config.n_channel == 1  # TODO
        # NOTE: DO NOT add bath normalization layer
        encoder_layers = [
            nn.Conv3d(config.n_channel, 8, (3, 3, 3), padding=1, stride=(2, 2, 2)),
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
            Reshape(-1, 64, 4, 4, 4),
            nn.ConvTranspose3d(64, 32, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, 4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 8, 4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(8, 1, (4, 4, 4), padding=1, stride=(2, 2, 2)),
        ]
        if config.output_binary:
            decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)


def create_pixel_encoder_decoder_112(
    n_channel: int, dim_bottleneck: int
) -> Tuple[nn.Sequential, nn.Sequential]:
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
        nn.Linear(128 * 16, dim_bottleneck),
        nn.ReLU(inplace=True),
    ]

    decoder_layers = [
        nn.Linear(dim_bottleneck, 128 * 16),
        nn.ReLU(inplace=True),
        Reshape(-1, 128, 4, 4),
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
    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


def create_pixel_encoder_decoder_56(
    n_channel: int, dim_bottleneck: int
) -> Tuple[nn.Sequential, nn.Sequential]:
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
        nn.Linear(1024, dim_bottleneck),
        nn.ReLU(inplace=True),
    ]

    decoder_layers = [
        nn.Linear(dim_bottleneck, 1024),
        nn.ReLU(inplace=True),
        Reshape(-1, 64, 4, 4),
        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(16, 4, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(4, 1, 4, stride=2, padding=1),
    ]
    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


class PixelAutoEncoder(ReconstructionLossMixin, NeuralAutoEncoderBase):
    def _setup_from_config(self, config: AutoEncoderConfig) -> None:
        n_channel = config.n_channel
        if config.n_grid == 112:
            encoder, decoder = create_pixel_encoder_decoder_112(n_channel, config.dim_bottleneck)
        elif config.n_grid == 56:
            encoder, decoder = create_pixel_encoder_decoder_56(n_channel, config.dim_bottleneck)
        else:
            raise ValueError("only 56 or 112 is supported")
        self.encoder = encoder
        self.decoder = decoder


# EXPERIMENTAL!!!
class ChannelSplitPixelAutoEncoder(ModelBase[AutoEncoderConfig], AutoEncoderBase):
    encoder_list: nn.ModuleList
    decoder_list: nn.ModuleList
    loss_called: bool = False  # flag to show the model is trained

    @property
    def trained(self) -> bool:
        return self.loss_called

    def get_device(self) -> torch.device:
        return self.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (N, C, H, W)
        Returns:
            Concatenated embedding of shape (N, C * dim_bottleneck)
        """
        # Encode each channel separately, then concatenate results
        # x[:, i : i+1, :, :] => shape (N, 1, H, W)
        encoded_list = []
        for i in range(self.config.n_channel):
            channel_i = x[:, i : i + 1, :, :]  # (N, 1, H, W)
            encoded_i = self.encoder_list[i](channel_i)  # (N, dim_bottleneck)
            encoded_list.append(encoded_i)

        encoded = torch.cat(encoded_list, dim=1)  # (N, C * dim_bottleneck)
        return encoded

    def loss(self, x: torch.Tensor) -> LossDict:
        self.loss_called = True
        d = {}
        for i in range(self.config.n_channel):
            channel_i = x[:, i : i + 1, :, :]
            encoded_i = self.encoder_list[i](channel_i)
            reconst_i = self.decoder_list[i](encoded_i)
            loss_i = nn.MSELoss()(channel_i, reconst_i)
            d[f"reconstruction_{i}"] = loss_i
        return LossDict(d)

    @property
    def n_bottleneck(self) -> int:
        return self.config.dim_bottleneck

    def _setup_from_config(self, config: AutoEncoderConfig) -> None:
        encoders, decoders = [], []
        for i in range(config.n_channel):
            if config.n_grid == 112:
                encoder, decoder = create_pixel_encoder_decoder_112(1, config.dim_bottleneck)
            elif config.n_grid == 56:
                encoder, decoder = create_pixel_encoder_decoder_56(1, config.dim_bottleneck)
            else:
                raise ValueError("only 56 or 112 is supported")
            encoders.append(encoder)
            decoders.append(decoder)
        self.encoder_list = nn.ModuleList(encoders)
        self.decoder_list = nn.ModuleList(decoders)


@dataclass
class CostPredictorWithEncoderConfig(ModelConfigBase):
    # A very bad design. but I don't have time. hahhahaha...
    costpred_model: CostPredictor
    ae_model: AutoEncoderBase

    def __post_init__(self):
        assert not isinstance(self.ae_model, NullAutoEncoder)


class CostPredictorWithEncoder(ModelBase[CostPredictorWithEncoderConfig]):
    costpred_model: CostPredictor
    ae_model: AutoEncoderBase

    def put_on_device(self, device: Optional[torch.device] = None):
        super().put_on_device(device)
        self.costpred_model.put_on_device(device)

        # NOTE: ae_model cannot be nullautoencoder thus we put type-ignore
        self.ae_model.put_on_device(device)  # type: ignore

    def _setup_from_config(self, config: CostPredictorWithEncoderConfig) -> None:
        self.costpred_model = config.costpred_model
        self.ae_model = config.ae_model

    def forward_multi_inner(self, mesh: Tensor, descs: Tensor) -> Tensor:
        # descs shares the same mesh
        n_inner, _ = descs.shape
        encoded = self.ae_model.forward(mesh)
        encoded_repeated = encoded.repeat((n_inner, 1))
        costs_pred, _ = self.costpred_model.forward((encoded_repeated, descs))
        return costs_pred

    def forward(self, sample: Tuple[Tensor, Tensor]) -> Tensor:
        meshes, descs = sample
        encoded = self.ae_model.forward(meshes)
        costs_pred, _ = self.costpred_model.forward((encoded, descs))
        costs_pred = costs_pred.flatten()
        return costs_pred

    def loss(self, sample: Tuple[Tensor, Tensor, Tensor, Tensor]) -> LossDict:
        meshes, descs, costs, weithtss = sample
        costs_pred = self.forward((meshes, descs))
        costs_pred = costs_pred.flatten()
        costs = costs.flatten()

        weihted_cost_loss = (costs_pred - costs) ** 2 * weithtss.flatten()
        cost_loss = torch.mean(weihted_cost_loss)
        dic = {"cost": cost_loss}
        return LossDict(dic)


@dataclass
class FusingCostPredictorConfig(ModelConfigBase):  # don't use
    dim_vector_descriptor: int
    n_grid: int


class FusingCostPredictor(ModelBase[FusingCostPredictorConfig]):  # don't use
    ae_model: AutoEncoderBase
    conv_layers1: nn.Sequential
    fusing_layers: nn.Sequential
    conv_layers2: nn.Sequential
    linear_layers: nn.Sequential

    def _setup_from_config(self, config: FusingCostPredictorConfig) -> None:
        assert config.n_grid == 56, "only 56 is supported now"
        conv_layers1 = nn.Sequential(
            *[
                nn.Conv2d(1, 8, 3, padding=1, stride=(2, 2)),  # 28 x 28
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
            ]
        )

        fusing_layers = nn.Sequential(
            *[
                nn.Linear(config.dim_vector_descriptor, 28 * 28),
                nn.BatchNorm1d(28 * 28),
                nn.ReLU(inplace=True),
            ]
        )

        conv_layer2 = nn.Sequential(
            *[
                nn.Conv2d(9, 16, 3, padding=1, stride=(2, 2)),  # 14 x 14
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),  # 64 x 4 x 4
            ]
        )

        linear_layers = nn.Sequential(
            *[
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(inplace=True),
                nn.Linear(100, 1),
            ]
        )

        self.conv_layers1 = conv_layers1
        self.fusing_layers = fusing_layers
        self.conv_layers2 = conv_layer2
        self.linear_layers = linear_layers

    def forward(self, sample: Tuple[Tensor, Tensor]) -> Tensor:
        image, vector = sample
        image = self.conv_layers1(image)
        vector = self.fusing_layers(vector)
        vector = vector.view(-1, 1, 28, 28)
        image = torch.cat([image, vector], dim=1)
        image = self.conv_layers2(image)
        return self.linear_layers(image).flatten()

    def loss(self, sample: Tuple[Tensor, Tensor, Tensor, Tensor]) -> LossDict:
        image, vector, costs, weight = sample
        costs_pred = self.forward((image, vector))
        costs_pred = costs_pred.flatten()
        costs = costs.flatten()

        weihted_cost_loss = (costs_pred - costs) ** 2 * weight
        cost_loss = torch.mean(weihted_cost_loss)
        dic = {"cost": cost_loss}
        return LossDict(dic)
