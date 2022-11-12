import torch

from hifuku.nerual import (
    IterationPredictor,
    IterationPredictorConfig,
    VoxelAutoEncoder,
    VoxelAutoEncoderConfig,
)


def test_network():
    ae_config = VoxelAutoEncoderConfig()
    ae = VoxelAutoEncoder(ae_config)

    n_sol_dim = 10
    model_conf = IterationPredictorConfig(
        12, ae_config.dim_bottleneck, n_sol_dim, use_solution_pred=True
    )
    model = IterationPredictor(model_conf)

    n_batch = 10
    n_pose_pre_scene = 5

    mesh = torch.zeros(n_batch, 1, 56, 56, 28)
    mesh_encoded = ae.encoder(mesh)
    descriptions = torch.zeros(n_batch, n_pose_pre_scene, 12)
    solutions = torch.zeros(n_batch, n_pose_pre_scene, n_sol_dim)
    nits = torch.zeros(n_batch, n_pose_pre_scene)

    sample = (mesh_encoded, descriptions, nits, solutions)

    model.loss(sample)


if __name__ == "__main__":
    test_network()
