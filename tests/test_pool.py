import numpy as np

from hifuku.pool import SimpleIteratorProblemPool
from hifuku.threedim.tabletop import TabletopPlanningProblem


def test_simple_pool():
    pool = SimpleIteratorProblemPool(TabletopPlanningProblem, 10)

    for _ in range(5):
        next(pool)

    pred1 = lambda p: p.get_descriptions()[0][1] > -np.inf  # noqa
    pool_pred = pool.make_predicated(pred1, 40)
    for _ in range(5):
        assert next(pool_pred) is not None

    pred2 = lambda p: p.get_descriptions()[0][1] > np.inf  # noqa
    pool_pred = pool.make_predicated(pred2, 40)
    for _ in range(5):
        assert next(pool_pred) is None
