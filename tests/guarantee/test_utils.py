import numpy as np

from hifuku.guarantee.utils import compute_real_itervals
from hifuku.threedim.tabletop import TabletopPlanningProblem


def test_compute_real_itervals():
    np.random.seed(0)
    prob_stan = TabletopPlanningProblem.create_standard()
    res = prob_stan.solve()[0]
    x_init = res.x

    n_problem = 10
    problems = [TabletopPlanningProblem.sample(1) for _ in range(n_problem)]
    maxiter = TabletopPlanningProblem.get_solver_config().maxiter

    itervals_mp = compute_real_itervals(problems, x_init, maxiter, n_process=4)
    itervals_sp = compute_real_itervals(problems, x_init, maxiter, n_process=1)
    assert len(itervals_mp) == n_problem
    assert itervals_mp == itervals_sp


if __name__ == "__main__":
    test_compute_real_itervals()
