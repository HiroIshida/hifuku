import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from skplan.solver.inverse_kinematics import IKConfig

from hifuku.threedim.tabletop import TabletopIKProblem
from hifuku.types import RawData


def test_rawdata_dump_and_load():
    problem = TabletopIKProblem.sample(n_pose=1)
    av_init = np.zeros(10)
    results = problem.solve(av_init)
    data = RawData.create(problem, results, av_init, IKConfig())

    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "hoge.npz"
        data.dump(fp)
        fpgz = Path(td) / "hoge.npz.gz"
        assert fpgz.exists()

        cmd = "gunzip {}".format(fpgz)
        subprocess.run(cmd, shell=True)
        data_again = RawData.load(fp)

    assert pickle.dumps(data) == pickle.dumps(data_again)


def test_rawdata_to_tensor():
    n_actual_problem = 10
    problem = TabletopIKProblem.sample(n_pose=n_actual_problem)
    av_init = np.zeros(10)
    results = problem.solve(av_init)
    data = RawData.create(problem, results, av_init, IKConfig())
    meshes, descriptions, nfevs, solutions = data.to_tensors()

    assert meshes.ndim == 4
    assert descriptions.ndim == 2
    assert nfevs.ndim == 1

    assert len(meshes) == 1
    assert len(descriptions) == n_actual_problem
    assert len(nfevs) == n_actual_problem


if __name__ == "__main__":
    test_rawdata_dump_and_load()
