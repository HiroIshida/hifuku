import pickle
import subprocess
import tempfile
from pathlib import Path

from rpbench.tabletop import TabletopBoxRightArmReachingTask
from skmp.solver.ompl_solver import OMPLSolverConfig, OMPLSolverResult

from hifuku.types import RawData


def test_rawdata_dump_and_load():
    n_desc = 10
    task = TabletopBoxRightArmReachingTask.sample(n_desc)
    results = [OMPLSolverResult(None, 1.0, 10) for _ in range(n_desc)]  # dummy
    data = RawData(task.export_table(), tuple(results), OMPLSolverConfig())

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
    results = [OMPLSolverResult(None, 1.0, 10) for _ in range(n_desc)]  # dummy
    data = RawData(task.export_table(), tuple(results), OMPLSolverConfig())

    meshes, descriptions, nits = data.to_tensors()

    assert meshes.ndim == 4
    assert descriptions.shape == (n_desc, 6 + 6)
    assert nits.shape == (n_desc,)


if __name__ == "__main__":
    # test_rawdata_dump_and_load()
    test_rawdata_to_tensor()
