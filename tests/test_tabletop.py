import numpy as np

from hifuku.tabletop import TabletopIKProblem, IKConfig


def test_exact_grid_conversion():
    for _ in range(3):
        problem = TabletopIKProblem.sample()

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
        np.testing.assert_almost_equal(vals_approx, vals_analytical, decimal=0.001)


def test_solve_problem():

    av_init = np.zeros(10)
    sample_count = 0
    for _ in range(10):
        while True:
            sample_count += 1
            config = IKConfig(clearance=0.03)
            problem = TabletopIKProblem.sample()
            res = problem.solve(av_init, config=config)
            if res.success:
                efkin, colkin = problem.setup_kinmaps()
                assert not colkin.is_colliding(problem.get_sdf(), res.x)
                break
    print("sample count {}".format(sample_count))
