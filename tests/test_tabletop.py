import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from hifuku.threedim.robot import setup_kinmaps
from hifuku.threedim.tabletop import (
    CachedProblemPool,
    TabletopIKProblem,
    TabletopMeshProblem,
    TabletopPlanningProblem,
    VoxbloxTabletopMeshProblem,
    VoxbloxTabletopPlanningProblem,
)


@dataclass
class SimplePredicate:
    threshold: float = 0.0

    def __call__(self, problem: TabletopIKProblem) -> bool:
        assert problem.n_problem() == 1
        pose = problem.target_pose_list[0]
        is_y_positive = pose.worldpos()[1] > self.threshold
        return is_y_positive


def test_exact_grid_conversion():
    for _ in range(3):
        problem = TabletopIKProblem.sample(n_pose=5)

        # create points
        grid = problem.grid_sdf.grid
        X, Y, Z = grid.get_meshgrid(indexing="ij")
        pts = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))

        # sdf
        approx_sdf = problem.get_sdf()
        analytical_sdf = problem.world.get_union_sdf()

        # compare
        vals_approx = approx_sdf(pts)
        vals_analytical = analytical_sdf(pts)
        np.testing.assert_almost_equal(vals_approx, vals_analytical, decimal=2)


def test_sample_problem():
    feasible_predicate = SimplePredicate()
    infeasible_predicate = SimplePredicate(np.inf)
    problem = TabletopIKProblem.sample(30, predicate=feasible_predicate, max_trial_factor=1000)
    assert problem is not None
    for desc in problem.get_descriptions():
        assert len(desc) == 12
        y = desc[2]
        assert y > 0

    problem = TabletopIKProblem.sample(30, predicate=infeasible_predicate)
    assert problem is None


def test_casting():
    prob1 = VoxbloxTabletopMeshProblem.sample(0)
    prob2 = prob1.cast_to(VoxbloxTabletopPlanningProblem)
    prob2._aux_gridsdf_cache is not None

    # ok
    for problem_type in [
        VoxbloxTabletopMeshProblem,
        VoxbloxTabletopPlanningProblem,
        VoxbloxTabletopPlanningProblem,
    ]:
        prob1.cast_to(problem_type)

    # not ok
    for problem_type in [TabletopMeshProblem, TabletopIKProblem, TabletopPlanningProblem]:
        with pytest.raises(TypeError):
            prob1.cast_to(problem_type)


def test_solve_problem():

    av_init = np.zeros(10)
    sample_count = 0
    for _ in range(10):
        while True:
            sample_count += 1
            problem = TabletopIKProblem.sample(n_pose=1)
            res = problem.solve(av_init)[0]
            if res.success:
                efkin, colkin = setup_kinmaps()
                assert not colkin.is_colliding(problem.get_sdf(), res.x)
                break
    print("sample count {}".format(sample_count))


def test_CachedProblemPool():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        n_problem = 20
        for _ in range(n_problem):
            mesh_problem = TabletopMeshProblem.sample(0)
            mesh_problem.dump(td_path / (str(uuid.uuid4()) + ".pkl"))
        pool = CachedProblemPool.load(TabletopPlanningProblem, TabletopMeshProblem, 5, td_path)
        pool.reset()
        probs = [p for p in pool]
        assert len(probs) == n_problem
        for p in probs:
            assert p.n_problem() == 5

        pools = pool.split(3)
        paths = []
        for p in pools:
            paths.extend(p.cache_path_list)
        assert len(set(paths)) == 20
