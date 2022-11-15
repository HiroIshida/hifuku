import subprocess
import tempfile
from pathlib import Path

import numpy as np
from skplan.solver.inverse_kinematics import IKConfig
from skplan.solver.optimization import PlannerConfig

from hifuku.threedim.tabletop import TabletopIKProblem
from hifuku.types import RawData


def test_rawdata_dump_and_load():
    problem = TabletopIKProblem.sample(n_pose=1)
    av_init = np.zeros(10)
    results = problem.solve(av_init)
    data = RawData.create(problem, results, IKConfig())

    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "hoge.npz"
        data.dump(fp)
        fpgz = Path(td) / "hoge.npz.gz"
        assert fpgz.exists()

        cmd = "gunzip {}".format(fpgz)
        subprocess.run(cmd, shell=True)
        data_again = RawData.load(fp)

    np.testing.assert_almost_equal(data.mesh, data_again.mesh, decimal=5)
    np.testing.assert_almost_equal(
        np.array(data.descriptions), np.array(data_again.descriptions), decimal=5
    )
    np.testing.assert_almost_equal(np.array(data.nits), np.array(data_again.nits), decimal=5)
    assert np.array(data.successes) == np.array(data_again.successes)


def test_rawdata_to_tensor():
    n_actual_problem = 10
    problem = TabletopIKProblem.sample(n_pose=n_actual_problem)
    av_init = np.zeros(10)
    results = problem.solve(av_init)
    data = RawData.create(problem, results, PlannerConfig())
    meshes, descriptions, nfevs, solutions = data.to_tensors()

    assert meshes.ndim == 4
    assert descriptions.ndim == 2
    assert nfevs.ndim == 1

    assert len(meshes) == 1
    assert len(descriptions) == n_actual_problem
    assert len(nfevs) == n_actual_problem
