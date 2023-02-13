import torch

from hifuku.domain import TBRR_SQP_DomainProvider
from hifuku.neuralnet import (
    AutoEncoderConfig,
    IterationPredictor,
    IterationPredictorConfig,
    VoxelAutoEncoder,
)


def test_network():
    device = torch.device("cpu")
    # test autoencoder
    ae_config = AutoEncoderConfig()
    ae = VoxelAutoEncoder(ae_config, device=device)

    # test iteration predictor
    n_sol_dim = 10
    model_conf = IterationPredictorConfig(
        12, ae_config.dim_bottleneck, n_sol_dim, use_solution_pred=False
    )
    model = IterationPredictor(model_conf, device=device)

    n_batch = 10

    mesh = torch.zeros(n_batch, 1, 56, 56, 28)
    mesh_encoded = ae.encoder(mesh)
    descriptions = torch.zeros(n_batch, 12)
    nits = torch.zeros(n_batch, 1)

    sample = (mesh_encoded, descriptions, nits)
    model.loss(sample)

    task_type = TBRR_SQP_DomainProvider.get_task_type()
    task_type.sample(1)


if __name__ == "__main__":
    test_network()
