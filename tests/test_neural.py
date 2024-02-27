from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from mohou.trainer import TrainCache, TrainConfig, train

from hifuku.datagen.batch_solver import MultiProcessBatchTaskSolver
from hifuku.domain import DummyDomain, DummyMeshDomain
from hifuku.neuralnet import (
    AutoEncoderConfig,
    CostPredictor,
    CostPredictorConfig,
    CostPredictorWithEncoder,
    CostPredictorWithEncoderConfig,
    PixelAutoEncoder,
    VoxelAutoEncoder,
    create_dataset_from_paramss_and_resultss,
)


def test_network():
    device = torch.device("cpu")
    # test autoencoder
    ae_config = AutoEncoderConfig()
    ae = VoxelAutoEncoder(ae_config, device=device)

    # test cost predictor
    conf = CostPredictorConfig(12, ae_config.dim_bottleneck, (10, 10, 10))
    model1 = CostPredictor(conf, device=device)
    assert len(model1.linears) == 3 * 3 + 1

    conf = CostPredictorConfig(12, ae_config.dim_bottleneck, (10, 10, 10), use_batch_norm=False)
    model2 = CostPredictor(conf, device=device)
    assert len(model2.linears) == 3 * 2 + 1

    conf = CostPredictorConfig(12, ae_config.dim_bottleneck, (5, 5, 5, 5))
    model3 = CostPredictor(conf, device=device)
    assert len(model3.linears) == 3 * 4 + 1

    for model in [model1, model2, model3]:
        n_batch = 10
        mesh = torch.zeros(n_batch, 1, 56, 56, 28)
        mesh_encoded = ae.encoder(mesh)
        descriptions = torch.zeros(n_batch, 12)
        nits = torch.zeros(n_batch, 1)
        weights = torch.ones(n_batch, 1)

        sample = (mesh_encoded, descriptions, nits, weights)
        model.loss(sample)


@lru_cache(maxsize=None)
def get_sol_tasks_and_resultss(domain):
    task_type = domain.task_type
    task_default = task_type.sample(1, True)
    # get a valid solution until it is found
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
    task_paramss = np.array([task_type.sample(n_inner).to_task_params() for _ in range(n_task)])
    batch_solver = MultiProcessBatchTaskSolver(domain.solver_type, domain.solver_config, task_type)
    resultss = batch_solver.solve_batch(task_paramss, [sol.traj] * n_task)
    return sol, task_paramss, resultss


def _test_dataset(domain, use_weight: bool, encode_image: bool):
    # for domain in [DummyMeshDomain, DummyDomain]:
    sol_tasks_and_resultss = get_sol_tasks_and_resultss(domain)
    device = torch.device("cpu")
    if domain == DummyMeshDomain:
        ae_config = AutoEncoderConfig(n_grid=56)
        ae = PixelAutoEncoder(ae_config, device=device)
    else:
        ae = None
    sol, task_paramss, resultss = sol_tasks_and_resultss

    n_data = len(task_paramss) * len(task_paramss[0])

    # for use_weight in [True, False]:
    if use_weight:
        weightss = torch.ones(len(task_paramss), len(task_paramss[0])) * 2.0
        w_expected = 2.0
    else:
        weightss = None
        w_expected = 1.0

    # for encode_image in [True, False]:
    if encode_image:
        dataset = create_dataset_from_paramss_and_resultss(
            task_paramss, resultss, domain.solver_config, domain.task_type, weightss, ae
        )
    else:
        dataset = create_dataset_from_paramss_and_resultss(
            task_paramss, resultss, domain.solver_config, domain.task_type, weightss, None
        )
    assert len(dataset) == n_data
    assert dataset.n_inner == 3
    mesh, desc, it, w = dataset[0]
    if domain == DummyMeshDomain:
        assert mesh.ndim == 1 if encode_image else 3
    else:
        assert mesh.numel() == 0
    assert len(desc.shape) == 1
    assert len(it.shape) == 0
    assert len(w.shape) == 0
    assert w.item() == w_expected

    dataset.add(dataset)
    assert len(dataset) == n_data * 2
    mesh, desc, it, w = dataset[0]
    if domain == DummyMeshDomain:
        assert len(mesh.shape) == 1 if encode_image else 3
    else:
        assert mesh.numel() == 0
    assert len(desc.shape) == 1
    assert len(it.shape) == 0
    assert len(w.shape) == 0
    assert w.item() == w_expected


@pytest.mark.parametrize(
    "domain, use_weight, encode_image",
    [
        (DummyMeshDomain, True, True),
        (DummyMeshDomain, True, False),
        (DummyMeshDomain, False, True),
        (DummyMeshDomain, False, False),
        (DummyDomain, True, True),
        (DummyDomain, True, False),
        (DummyDomain, False, True),
        (DummyDomain, False, False),
    ],
)
def test_dataset(domain, use_weight, encode_image):
    _test_dataset(domain, use_weight, encode_image)


def _test_training(domain, use_pretrained_ae: bool):
    domain = DummyMeshDomain
    sol_tasks_and_resultss = get_sol_tasks_and_resultss(domain)
    device = torch.device("cpu")
    ae_config = AutoEncoderConfig(n_grid=56)
    ae = PixelAutoEncoder(ae_config, device=device)
    sol, task_paramss, resultss = sol_tasks_and_resultss

    train_config = TrainConfig(5, n_epoch=2)
    n_dof_desc = 2

    conf = CostPredictorConfig(n_dof_desc, ae_config.dim_bottleneck, (10, 10, 10))
    costpred_model = CostPredictor(conf, device=device)

    if use_pretrained_ae:
        dataset = create_dataset_from_paramss_and_resultss(
            task_paramss, resultss, domain.solver_config, domain.task_type, None, ae
        )
        model = costpred_model
    else:
        dataset = create_dataset_from_paramss_and_resultss(
            task_paramss, resultss, domain.solver_config, domain.task_type, None, None
        )
        conf = CostPredictorWithEncoderConfig(costpred_model, ae)
        model = CostPredictorWithEncoder(conf, device=device)

    with TemporaryDirectory() as td:
        td_path = Path(td)
        tcache = TrainCache.from_model(model)
        train(td_path, tcache, dataset, train_config)


@pytest.mark.parametrize(
    "domain, use_pretrained_ae",
    [(DummyMeshDomain, True), (DummyMeshDomain, False), (DummyDomain, True), (DummyDomain, False)],
)
def tset_training(domain, use_pretrained_ae):
    _test_training(domain, use_pretrained_ae)
