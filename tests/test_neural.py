from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from mohou.trainer import TrainCache, TrainConfig, train

from hifuku.datagen.batch_solver import MultiProcessBatchProblemSolver
from hifuku.domain import BubblySimpleMeshPointConnecting_RRT_Domain
from hifuku.neuralnet import (
    AutoEncoderConfig,
    IterationPredictor,
    IterationPredictorConfig,
    IterationPredictorDataset,
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

        sample = (mesh_encoded, descriptions, nits)
        model.loss(sample)


def test_train():
    device = torch.device("cpu")
    ae_config = AutoEncoderConfig()
    ae = PixelAutoEncoder(ae_config, device=device)

    domain = BubblySimpleMeshPointConnecting_RRT_Domain
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
    tasks = [task_type.sample(2) for _ in range(n_task)]
    batch_solver = MultiProcessBatchProblemSolver(domain.solver_type, domain.solver_config)
    resultss = batch_solver.solve_batch(tasks, [sol.traj] * n_task)

    dataset = IterationPredictorDataset.construct_from_tasks_and_resultss(
        sol.traj, tasks, resultss, domain.solver_config, ae
    )

    n_dof_desc = 4
    conf = IterationPredictorConfig(n_dof_desc, ae_config.dim_bottleneck, 10, 10, 10)
    iterpred_model = IterationPredictor(conf, device=device)

    train_config = TrainConfig(5, n_epoch=2)
    with TemporaryDirectory() as td:
        td_path = Path(td)
        tcache = TrainCache.from_model(iterpred_model)
        train(td_path, tcache, dataset, train_config)


if __name__ == "__main__":
    test_train()
