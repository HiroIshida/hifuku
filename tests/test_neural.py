from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from mohou.trainer import TrainCache, TrainConfig, train

from hifuku.datagen.batch_solver import MultiProcessBatchProblemSolver
from hifuku.domain import DoubleIntegratorBubblySimple_SQP
from hifuku.neuralnet import (
    AutoEncoderConfig,
    IterationPredictor,
    IterationPredictorConfig,
    IterationPredictorDataset,
    IterationPredictorWithEncoder,
    IterationPredictorWithEncoderConfig,
    PixelAutoEncoder,
    VoxelAutoEncoder,
)


def test_network():
    device = torch.device("cpu")
    # test autoencoder
    ae_config = AutoEncoderConfig()
    ae = VoxelAutoEncoder(ae_config, device=device)

    # test iteration predictor
    conf = IterationPredictorConfig(
        12, ae_config.dim_bottleneck, 10, 10, 10, use_solution_pred=False
    )
    model1 = IterationPredictor(conf, device=device)
    assert len(model1.linears) == 7

    conf = IterationPredictorConfig(
        12, ae_config.dim_bottleneck, layers=[5, 5, 5, 5], use_solution_pred=False
    )
    model2 = IterationPredictor(conf, device=device)
    assert len(model2.linears) == 9

    for model in [model1, model2]:
        n_batch = 10
        mesh = torch.zeros(n_batch, 1, 56, 56, 28)
        mesh_encoded = ae.encoder(mesh)
        descriptions = torch.zeros(n_batch, 12)
        nits = torch.zeros(n_batch, 1)
        weights = torch.ones(n_batch, 1)

        sample = (mesh_encoded, descriptions, nits, weights)
        model.loss(sample)


@pytest.fixture(autouse=True, scope="module")
def sol_tasks_and_resultss():
    domain = DoubleIntegratorBubblySimple_SQP
    task_type = domain.task_type

    task_default = task_type.sample(1, True)
    trial_count = 0
    while True:
        sol = task_default.solve_default()[0]
        if sol.traj is not None:
            break
        trial_count += 1
        if trial_count > 10:
            assert False

    n_task = 20
    n_inner = 3
    tasks = [task_type.sample(n_inner) for _ in range(n_task)]
    batch_solver = MultiProcessBatchProblemSolver(domain.solver_type, domain.solver_config)
    resultss = batch_solver.solve_batch(tasks, [sol.traj] * n_task)
    return sol, tasks, resultss


def test_dataset(sol_tasks_and_resultss):
    domain = DoubleIntegratorBubblySimple_SQP
    device = torch.device("cpu")
    ae_config = AutoEncoderConfig(n_grid=56)
    ae = PixelAutoEncoder(ae_config, device=device)
    sol, tasks, resultss = sol_tasks_and_resultss

    n_data = len(tasks) * tasks[0].n_inner_task

    for use_weight in [True, False]:
        if use_weight:
            weightss = torch.ones(len(tasks), tasks[0].n_inner_task) * 2.0
            w_expected = 2.0
        else:
            weightss = None
            w_expected = 1.0

        # test dataset when ae is specified (encoded)
        dataset = IterationPredictorDataset.construct_from_tasks_and_resultss(
            sol.traj, tasks, resultss, domain.solver_config, weightss, ae
        )
        assert len(dataset) == n_data
        assert dataset.n_inner == 3
        mesh, desc, it, w = dataset[0]
        assert len(mesh.shape) == 1
        assert len(desc.shape) == 1
        assert len(it.shape) == 0
        assert len(w.shape) == 0
        assert w.item() == w_expected

        dataset.add(dataset)
        assert len(dataset) == n_data * 2
        mesh, desc, it, w = dataset[0]
        assert len(mesh.shape) == 1
        assert len(desc.shape) == 1
        assert len(it.shape) == 0
        assert len(w.shape) == 0
        assert w.item() == w_expected

        # test dataset when ae is not specified
        dataset = IterationPredictorDataset.construct_from_tasks_and_resultss(
            sol.traj, tasks, resultss, domain.solver_config, weightss, None
        )
        assert len(dataset) == n_data
        assert dataset.n_inner == 3
        mesh, desc, it, w = dataset[0]
        assert len(mesh.shape) == 3
        assert len(desc.shape) == 1
        assert len(it.shape) == 0
        assert len(w.shape) == 0
        assert w.item() == w_expected

        dataset.add(dataset)
        assert len(dataset) == n_data * 2
        mesh, desc, it, w = dataset[0]
        assert len(mesh.shape) == 3
        assert len(desc.shape) == 1
        assert len(it.shape) == 0
        assert len(w.shape) == 0
        assert w.item() == w_expected


def test_training(sol_tasks_and_resultss):
    domain = DoubleIntegratorBubblySimple_SQP
    device = torch.device("cpu")
    ae_config = AutoEncoderConfig(n_grid=56)
    ae = PixelAutoEncoder(ae_config, device=device)
    sol, tasks, resultss = sol_tasks_and_resultss

    train_config = TrainConfig(5, n_epoch=2)
    n_dof_desc = 4

    # test dataset when ae is specified (encoded)
    dataset = IterationPredictorDataset.construct_from_tasks_and_resultss(
        sol.traj, tasks, resultss, domain.solver_config, None, ae
    )

    conf = IterationPredictorConfig(n_dof_desc, ae_config.dim_bottleneck, 10, 10, 10)
    iterpred_model = IterationPredictor(conf, device=device)

    with TemporaryDirectory() as td:
        td_path = Path(td)
        tcache = TrainCache.from_model(iterpred_model)
        train(td_path, tcache, dataset, train_config)

    # test dataset when ae is not specified
    dataset_raw = IterationPredictorDataset.construct_from_tasks_and_resultss(
        sol.traj, tasks, resultss, domain.solver_config, None, None
    )
    conf = IterationPredictorWithEncoderConfig(iterpred_model, ae)
    combined_model = IterationPredictorWithEncoder(conf, device=device)

    with TemporaryDirectory() as td:
        td_path = Path(td)
        tcache = TrainCache.from_model(combined_model)
        train(td_path, tcache, dataset_raw, train_config)


if __name__ == "__main__":
    # test_training(sol_tasks_and_resultss())
    pass
