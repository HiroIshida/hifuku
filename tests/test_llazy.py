import hashlib
import os
import pickle
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from hifuku.llazy.dataset import (
    LazyDecomplessDataLoader,
    LazyDecomplessDataset,
    PicklableChunkBase,
)


@dataclass
class ExampleChunk(PicklableChunkBase):
    data: np.ndarray

    def to_tensors(self) -> torch.Tensor:
        a = torch.from_numpy(self.data).float()
        return a


@dataclass
class ExampleChunk2(PicklableChunkBase):
    data: np.ndarray

    def to_tensors(self) -> Tuple[torch.Tensor, ...]:
        a = torch.from_numpy(self.data).float()
        return a, a


def test_dataset():
    def compute_hashint(chunk: ExampleChunk) -> int:
        byte = hashlib.md5(pickle.dumps(chunk)).digest()
        return int.from_bytes(byte, "big", signed=True)

    with tempfile.TemporaryDirectory() as td:
        base_path = Path(td)

        # prepare chunks
        hash_value_original = 0
        n_chunk = 200
        for _ in range(n_chunk):
            data = np.random.randn(56, 56, 28)
            chunk = ExampleChunk(data)
            name = str(uuid.uuid4()) + ".pkl"
            path = base_path / name
            chunk.dump(path)
            hash_value_original += compute_hashint(chunk)

        # load
        elapsed_times = []
        for n_worker in [1, 2]:
            dataset = LazyDecomplessDataset.load(base_path, ExampleChunk, n_worker=n_worker)

            hash_value_load = 0

            ts = time.time()
            indices_per_chunks = np.array_split(np.array(range(n_chunk)), 3)
            for indices in indices_per_chunks:
                chunks = dataset.get_data(indices)
                assert len(chunks) == len(indices), "{} <-> {}".format(len(chunks), len(indices))

                for chunk in chunks:
                    hash_value_load += compute_hashint(chunk)
            elapsed_times.append(time.time() - ts)

            trash_num = len([p for p in base_path.iterdir() if p.name.endswith(".pkl")])
            assert trash_num == 0

            # compare hash
            assert hash_value_original == hash_value_load

            # test random split
            train_dataset, valid_dataset = dataset.random_split(0.1)
            assert len(train_dataset) == 180
            assert len(valid_dataset) == 20
            set_before = set(dataset.compressed_path_list)
            set_after = set(train_dataset.compressed_path_list + valid_dataset.compressed_path_list)
            assert set_before == set_after

        cpu_count = os.cpu_count()
        assert cpu_count is not None
        has_morethan_two_core = cpu_count >= 4
        if has_morethan_two_core:
            print(elapsed_times)
            assert elapsed_times[1] < elapsed_times[0] * 0.9


def test_dataset_iterator():

    for chunk_size in [3, 10, 13]:
        with tempfile.TemporaryDirectory() as td:
            base_path = Path(td)

            # prepare chunks
            for _ in range(chunk_size):
                data = np.random.randn(2, 3, 4)
                chunk = ExampleChunk(data)
                name = str(uuid.uuid4()) + ".pkl"
                path = base_path / name
                chunk.dump(path)

            dataset = LazyDecomplessDataset.load(base_path, ExampleChunk, n_worker=2)  # type: ignore
            for e in dataset:
                pass


def test_dataloader():

    with tempfile.TemporaryDirectory() as td:
        base_path = Path(td)

        # prepare chunks
        n_chunk = 120
        for _ in range(n_chunk):
            data = np.random.randn(2, 3, 4)
            chunk = ExampleChunk(data)
            name = str(uuid.uuid4()) + ".pkl"
            path = base_path / name
            chunk.dump(path)

        for batch_size in [20, 50, 80, 100, 120]:  # try to hit edge cases
            for chunk_t in [ExampleChunk, ExampleChunk2]:
                dataset = LazyDecomplessDataset.load(base_path, chunk_t, n_worker=2)  # type: ignore
                loader = LazyDecomplessDataLoader(dataset, batch_size=batch_size)

                index_set = set()  # type: ignore
                loader.__iter__()  # initialize
                assert loader._indices_per_iter is not None
                for indices in loader._indices_per_iter:
                    index_set = index_set.union(indices)
                assert index_set == set(list(range(n_chunk)))

                n_data_sum = 0
                for sample in loader:
                    if chunk_t == ExampleChunk2:
                        assert len(sample) == 2
                        sample = sample[0]

                    n_data_sum += len(sample)
                    assert isinstance(sample, torch.Tensor)
                    assert sample.dim() == 4
                    n_batch, a, b, c = sample.shape
                    assert n_batch in (batch_size, n_chunk % batch_size)
                    assert a == 2
                    assert b == 3
                    assert c == 4

                assert n_data_sum == n_chunk


if __name__ == "__main__":
    # test_dataset()
    test_dataloader()
