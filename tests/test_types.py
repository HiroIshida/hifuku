import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from skmp.solver.ompl_solver import OMPLSolverConfig, OMPLSolverResult, TerminateState
from skmp.trajectory import Trajectory

from hifuku.rpbench_wrap import TabletopBoxRightArmReachingTask
from hifuku.types import RawData


def test_rawdata_dump_and_load():
    n_desc = 10
    task = TabletopBoxRightArmReachingTask.sample(n_desc)
    results = [
        OMPLSolverResult(None, 1.0, 10, TerminateState.SUCCESS) for _ in range(n_desc)
    ]  # dummy
    traj_dummy = Trajectory([np.zeros(2)])
    data = RawData(traj_dummy, task.export_table(), tuple(results), OMPLSolverConfig())

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
    task = TabletopBoxRightArmReachingTask.sample(n_desc)
    traj_dummy = Trajectory([np.zeros(2)])
    results = [
        OMPLSolverResult(None, 1.0, 10, TerminateState.SUCCESS) for _ in range(n_desc)
    ]  # dummy
    data = RawData(traj_dummy, task.export_table(), tuple(results), OMPLSolverConfig())

    meshes, descriptions, nits = data.to_tensors()

    assert meshes.ndim == 4
    assert descriptions.shape == (n_desc, 6 + 6)
    assert nits.shape == (n_desc,)


def test_rawdata_to_tensor_iternum_clamp():
    task = TabletopBoxRightArmReachingTask.sample(4)
    traj_dummy = Trajectory([np.zeros(2)])
    n_max_call = 5
    results = [
        OMPLSolverResult(traj_dummy, 1.0, 1, TerminateState.FAIL_PLANNING),
        OMPLSolverResult(traj_dummy, 1.0, 5, TerminateState.FAIL_PLANNING),
        OMPLSolverResult(None, 1.0, 1, TerminateState.SUCCESS),
        OMPLSolverResult(None, 1.0, 5, TerminateState.SUCCESS),
    ]  # dummy
    data = RawData(
        traj_dummy, task.export_table(), tuple(results), OMPLSolverConfig(n_max_call=n_max_call)
    )
    _, _, torch_nits = data.to_tensors()
    assert list(torch_nits.numpy()) == [1, 5, n_max_call + 1, n_max_call + 1]


if __name__ == "__main__":
    # test_rawdata_dump_and_load()
    test_rawdata_to_tensor()
