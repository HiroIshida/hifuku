import torch

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


if __name__ == "__main__":
    test_network()
