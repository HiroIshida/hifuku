from rpbench.two_dimensional.dummy import DummyTask

from hifuku.pool import TaskPool


def test_simple_pool():
    pool = TaskPool(DummyTask)

    for _ in range(5):
        next(pool)

    pool_pred = pool.make_predicated(lambda x: True, 40)
    for _ in range(5):
        assert next(pool_pred) is not None

    pool_pred = pool.make_predicated(lambda x: False, 40)
    pool_pred.timeout = 0.2
    for _ in range(5):
        assert next(pool_pred) is None
