import copy
import os
import pickle
import random
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch
from torch.utils.data import default_collate

_has_gzip = subprocess.run("which gzip > /dev/null", shell=True).returncode == 0
_has_pigz = subprocess.run("which pigz > /dev/null", shell=True).returncode == 0
assert _has_gzip or _has_pigz

if _has_pigz:
    _zip_command = "pigz"
    _unzip_command = "unpigz"
else:
    _zip_command = "gzip"
    _unzip_command = "gunzip"

ChunkT = TypeVar("ChunkT", bound="ChunkBase")
PicklableChunkT = TypeVar("PicklableChunkT", bound="PicklableChunkBase")


class ChunkBase(ABC):
    @classmethod
    @abstractmethod
    def load(cls: Type[ChunkT], path: Path) -> ChunkT:
        pass

    @abstractmethod
    def dump_impl(self, path: Path) -> None:
        pass

    @abstractmethod
    def to_tensors(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        pass

    def dump(self, path: Path) -> Path:
        self.dump_impl(path)
        command = "{0} -1 {1}".format(_zip_command, path)
        subprocess.run(command, shell=True)
        return path.parent / (path.name + ".gz")


@dataclass  # type: ignore
class PicklableChunkBase(ChunkBase):
    @classmethod
    def load(cls: Type[PicklableChunkT], path: Path) -> PicklableChunkT:
        with path.open(mode="rb") as f:
            dic = pickle.load(f)
        return cls(**dic)

    def dump_impl(self, path: Path) -> None:
        # as a shallow dict instead of dataclasses.asdict
        dic = {}
        for field in fields(self):
            key = field.name
            dic[key] = self.__dict__[key]
        with path.open(mode="wb") as f:
            pickle.dump(dic, f)


@dataclass
class LazyDecomplessDataset(Iterable[ChunkT], Generic[ChunkT]):
    base_path: Path
    compressed_path_list: List[Path]
    chunk_type: Type[ChunkT]
    n_worker: int
    parallelize_threshold: int = 20

    def __post_init__(self):
        if self.n_worker == -1:
            n_cpu_virtual = os.cpu_count()
            assert n_cpu_virtual is not None
            n_cpu = int(round(n_cpu_virtual * 0.5))
            self.n_worker = n_cpu

    @classmethod
    def load(
        cls, base_path: Path, chunk_type: Type[ChunkT], n_worker: int = -1
    ) -> "LazyDecomplessDataset[ChunkT]":
        path_list = []
        for p in base_path.iterdir():
            if p.name.endswith(".gz"):
                path_list.append(p)
        return cls(base_path, path_list, chunk_type, n_worker)

    def __len__(self) -> int:
        return len(self.compressed_path_list)

    def get_data(self, indices: np.ndarray) -> List[ChunkT]:
        command = "xargs -n 1 -P {0} {1} --keep --force".format(self.n_worker, _unzip_command)
        paths_compressed = [self.compressed_path_list[i] for i in indices]
        paths_compressed_as_byte = "\n".join([str(p) for p in paths_compressed]).encode()
        subprocess.run(command, shell=True, input=paths_compressed_as_byte, text=False, check=True)

        def index_to_uncompressed_path(idx: int) -> Path:
            path = self.compressed_path_list[idx]
            path_decompressed = path.parent / path.stem
            return path_decompressed

        paths = [index_to_uncompressed_path(i) for i in indices]
        chunk_list = [self.chunk_type.load(path) for path in paths]

        # remove trash
        paths_as_byte = "\n".join([str(p) for p in paths]).encode()
        subprocess.run("xargs rm", shell=True, input=paths_as_byte, text=False, check=True)
        return chunk_list

    @staticmethod
    def decompress(paths: List[Path]) -> None:
        command = _unzip_command
        string_paths = [str(path) for path in paths]
        command += " -f --keep %s" % " ".join(string_paths)
        subprocess.run(command, shell=True)

    def random_split(
        self, valid_rate: float = 0.1
    ) -> Tuple["LazyDecomplessDataset", "LazyDecomplessDataset"]:
        path_list = copy.deepcopy(self.compressed_path_list)
        random.shuffle(path_list)

        n_valid = int(len(self) * valid_rate)
        dataset_train = copy.deepcopy(self)
        dataset_train.compressed_path_list = path_list[:-n_valid]
        dataset_valid = copy.deepcopy(self)
        dataset_valid.compressed_path_list = path_list[-n_valid:]
        return dataset_train, dataset_valid

    def __iter__(self) -> Iterator[ChunkT]:
        return DatasetIterator(self)


@dataclass
class DatasetIterator(Iterator[ChunkT], Generic[ChunkT]):
    dataset: LazyDecomplessDataset
    _idx: int = 0

    def __post_init__(self):
        assert self._idx == 0

    def __next__(self):
        if self._idx == len(self.dataset):
            raise StopIteration
        data = self.dataset.get_data(np.array([self._idx]))[0]
        self._idx += 1
        return data


@dataclass
class LazyDecomplessDataLoader(Generic[ChunkT]):
    r"""
    A dataloader class that has the similar inteface as
    pytorch DataLoader
    """
    dataset: LazyDecomplessDataset[ChunkT]
    batch_size: int = 1
    shuffle: bool = True
    max_data_num: Optional[int] = None
    _indices_per_iter: Optional[List[np.ndarray]] = None  # set when __iter__ called

    def __len__(self) -> int:
        n_dataset = len(self.dataset)
        return n_dataset // self.batch_size + int(n_dataset % self.batch_size > 0)

    def __iter__(self) -> "LazyDecomplessDataLoader[ChunkT]":
        n_dataset = len(self.dataset)
        if self.max_data_num is not None:
            n_dataset = min(n_dataset, self.max_data_num)

        assert n_dataset > 0
        indices = np.arange(n_dataset)
        if self.shuffle:
            np.random.shuffle(indices)

        indices_list_per_iter = []
        head = 0
        for i in range(n_dataset // self.batch_size):
            indices = np.arange(head, head + self.batch_size)
            indices_list_per_iter.append(indices)
            head += self.batch_size

        rem = n_dataset % self.batch_size
        if rem > 0:
            indices = np.arange(head, head + rem)
            indices_list_per_iter.append(indices)

        self._indices_per_iter = indices_list_per_iter
        return self

    def __next__(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        assert self._indices_per_iter is not None
        if len(self._indices_per_iter) == 0:
            raise StopIteration()

        indices = self._indices_per_iter.pop()
        chunk_list = self.dataset.get_data(indices)

        zipped = [chunk.to_tensors() for chunk in chunk_list]
        return default_collate(zipped)
