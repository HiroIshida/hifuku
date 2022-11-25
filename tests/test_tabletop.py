from dataclasses import dataclass

import numpy as np

from hifuku.threedim.tabletop import TabletopIKProblem


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
    problem = TabletopIKProblem.sample(30, predicate=feasible_predicate)
    assert problem is not None
    for desc in problem.get_descriptions():
        assert len(desc) == 12
        y = desc[2]
        assert y > 0

    problem = TabletopIKProblem.sample(30, predicate=infeasible_predicate)
    assert problem is None


def test_solve_problem():

    av_init = np.zeros(10)
    sample_count = 0
    for _ in range(10):
        while True:
            sample_count += 1
            problem = TabletopIKProblem.sample(n_pose=1)
            res = problem.solve(av_init)[0]
            if res.success:
                efkin, colkin = problem.setup_kinmaps()
                assert not colkin.is_colliding(problem.get_sdf(), res.x)
                break
    print("sample count {}".format(sample_count))
