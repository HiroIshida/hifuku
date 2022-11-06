import numpy as np

from hifuku.sdf import create_union_sdf
from hifuku.tabletop import TabletopIKProblem


def test_exact_grid_conversion():
    for _ in range(3):
        problem = TabletopIKProblem.sample()

        # create points
        grid = problem.grid_sdf.grid
        X, Y, Z = grid.get_meshgrid(indexing="ij")
        pts = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))

        # sdf
        approx_sdf = create_union_sdf((problem.grid_sdf, problem.world.table.sdf))  # type: ignore
        analytical_sdf = problem.world.get_union_sdf()

        # compare
        vals_approx = approx_sdf(pts)
        vals_analytical = analytical_sdf(pts)
        np.testing.assert_almost_equal(vals_approx, vals_analytical)
