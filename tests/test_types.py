import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from rpbench.two_dimensional.dummy import (
    DummyConfig,
    DummyMeshTask,
    DummyResult,
    DummyTask,
)
from skmp.trajectory import Trajectory

from hifuku.types import RawData


def test_rawdata_dump_and_load():
    n_desc = 10
    task = DummyMeshTask.sample(n_desc)
    results = [DummyResult(None, 1.0, 10) for _ in range(n_desc)]
    traj_dummy = Trajectory([np.zeros(2)])
    data = RawData(traj_dummy, task.export_table(), tuple(results), DummyConfig(5))

    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "hoge.pkl"
        data.dump(fp)
        fpgz = Path(td) / "hoge.pkl.gz"
        assert fpgz.exists()

        cmd = "gunzip {}".format(fpgz)
        subprocess.run(cmd, shell=True)
        data_again = RawData.load(fp)

    assert pickle.dumps(data) == pickle.dumps(data_again)


def test_rawdata_to_tensor():
    n_desc = 10
    for task_type in [DummyMeshTask, DummyTask]:
        task = task_type.sample(n_desc)
        results = [DummyResult(None, 1.0, 10) for _ in range(n_desc)]
        traj_dummy = Trajectory([np.zeros(2)])
        data = RawData(traj_dummy, task.export_table(), tuple(results), DummyConfig(5))

        meshes, descriptions, nits, failed = data.to_tensors()

        if task_type == DummyMeshTask:
            assert meshes.ndim == 3
        else:
            assert meshes.shape == (0,)
        assert descriptions.shape == (n_desc, 2)
        assert nits.shape == (n_desc,)
        assert failed.shape == (n_desc,)


def test_rawdata_to_tensor_iternum_clamp():
    task = DummyMeshTask.sample(4)
    traj_dummy = Trajectory([np.zeros(2)])
    n_max_call = 5
    results = [
        DummyResult(traj_dummy, 1.0, 1),
        DummyResult(traj_dummy, 1.0, 5),
        DummyResult(None, 1.0, 1),
        DummyResult(None, 1.0, 5),
    ]  # dummy
    data = RawData(traj_dummy, task.export_table(), tuple(results), DummyConfig(n_max_call))
    _, _, torch_nits, _ = data.to_tensors()
    assert list(torch_nits.numpy()) == [1, 5, n_max_call * 2, n_max_call * 2]


if __name__ == "__main__":
    # test_rawdata_dump_and_load()
    test_rawdata_to_tensor()
