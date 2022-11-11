import torch

from hifuku.nerual import IterationPredictor, IterationPredictorConfig


def test_network():
    n_sol_dim = 10
    model_conf = IterationPredictorConfig(
        6, n_sol_dim, use_solution_pred=True, use_reconstruction=True
    )
    model = IterationPredictor(model_conf)

    n_batch = 10
    n_pose_pre_scene = 5

    mesh = torch.zeros(n_batch, 1, 56, 56, 28)
    descriptions = torch.zeros(n_batch, n_pose_pre_scene, 6)
    solutions = torch.zeros(n_batch, n_pose_pre_scene, n_sol_dim)
    nits = torch.zeros(n_batch, n_pose_pre_scene)

    sample = (mesh, descriptions, nits, solutions)

    model.loss(sample)


if __name__ == "__main__":
    test_network()
