import argparse
import multiprocessing
import resource
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import torch
import yaml
from mohou.trainer import TrainConfig

from hifuku.core import LibrarySamplerConfig, SimpleSolutionLibrarySampler
from hifuku.domain import select_domain
from hifuku.neuralnet import NullAutoEncoder, PixelAutoEncoder
from hifuku.script_utils import (
    create_default_logger,
    filter_warnings,
    get_project_path,
    load_compatible_autoencoder,
    watch_memory,
)


def parse_config_yaml(dic: Dict) -> LibrarySamplerConfig:
    if "train_config" in dic:
        train_config = TrainConfig(**dic["train_config"])
        dic["train_config"] = train_config
    ls_config = LibrarySamplerConfig(**dic)
    return ls_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str, default="tbrr_sqp", help="")
    parser.add_argument("-n", type=int, default=100, help="")
    parser.add_argument("-n_limit_batch", type=int, help="", default=50000)
    parser.add_argument("-conf", type=str)
    parser.add_argument("-post", type=str)
    parser.add_argument("--warm", action="store_true", help="warm start")
    parser.add_argument("--untrained", action="store_true", help="use untrained autoencoder")
    parser.add_argument("--local", action="store_true", help="don't use distributed computers")

    args = parser.parse_args()
    domain_name: str = args.type
    warm_start: bool = args.warm
    n_step: int = args.n
    use_distributed: bool = not args.local
    use_pretrained_ae: bool = not args.untrained
    library_sampling_conf_path_str: Optional[str] = args.conf
    project_name_postfix: Optional[str] = args.post

    # require explicitly setting to None
    assert project_name_postfix is not None
    if project_name_postfix == "none":
        project_name_postfix = None

    torch.backends.cudnn.enabled = False
    assert torch.cuda.is_available()

    filter_warnings()
    domain = select_domain(domain_name)
    project_path = get_project_path(domain_name, project_name_postfix)

    logger = create_default_logger(project_path, "library_gen")

    logger.info("cmd args: {}".format(args))

    # set file open limit to large
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    logger.info("current NOFILE value: (soft: {}, hard: {})".format(soft, hard))
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    logger.info("set NOFILE to {}".format(hard))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    logger.info("current NOFILE value: (soft: {}, hard: {})".format(soft, hard))

    if library_sampling_conf_path_str is None:
        library_sampling_conf_path = project_path / "lsconf.yaml"
    else:
        library_sampling_conf_path = Path(library_sampling_conf_path_str).expanduser()
    with library_sampling_conf_path.open(mode="r") as f:
        dic = yaml.safe_load(f)
        lsconfig = parse_config_yaml(dic)
        if not use_pretrained_ae:
            assert lsconfig.train_with_encoder
    logger.info("lsconfig: {}".format(lsconfig))

    if not use_pretrained_ae:
        assert (
            lsconfig.train_with_encoder
        ), "you must train encoder, otherwise encoder will be just a random one"

    # run memory watchdog
    p_watchdog = multiprocessing.Process(target=watch_memory, args=(5.0,))
    p_watchdog.start()

    n_grid = None
    if not use_pretrained_ae:
        task_type = domain.task_type
        task = task_type.sample()
        mat = task.export_task_expression(use_matrix=True).get_matrix()
        n_grid: Optional[Literal[56, 112]] = mat.shape[0]
    ae_model = load_compatible_autoencoder(domain_name, use_pretrained_ae, n_grid)

    if not isinstance(ae_model, NullAutoEncoder) and use_pretrained_ae:
        # check if the autoencoder is properly trained
        task = domain.task_type.sample()
        expression = task.export_task_expression(use_matrix=True)
        matrix = expression.get_matrix()
        assert matrix is not None
        world_mat_np = np.expand_dims(matrix, axis=(0, 1)).astype(float)
        world_mat_torch = torch.from_numpy(world_mat_np).float().to(ae_model.get_device())
        assert isinstance(ae_model, PixelAutoEncoder)
        decoded = ae_model.decoder(ae_model.encoder(world_mat_torch))
        mat_reconstructed = decoded.detach().cpu().squeeze().numpy()

        # compare mesh and mesh_reconstructed side by side in matplotlib
        # import matplotlib.pyplot as plt

        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # n_grid = ae_model.config.n_grid
        # axes[0].imshow(mesh_np.reshape(n_grid, n_grid))
        # axes[1].imshow(mesh_reconstructed.reshape(n_grid, n_grid))

        # def close_plot_after_timeout(timeout):
        #     plt.pause(timeout)
        #     plt.close()

        # # close automatically after 5 seconds
        # t = threading.Thread(target=close_plot_after_timeout, args=(4,))
        # t.start()
        # plt.show()

    lib_sampler = SimpleSolutionLibrarySampler.initialize(
        domain.task_type,
        domain.solver_type,
        domain.solver_config,
        ae_model,
        lsconfig,
        project_path,
        use_distributed=use_distributed,
        n_limit_batch_solver=args.n_limit_batch,
        warm_start=warm_start,
    )

    for i in range(n_step):
        profs = lib_sampler.sampler_history.elapsed_time_history
        if len(profs) > 0:
            time_total = sum([info.t_total for info in profs])  # type: ignore[misc]
            logger.info("time_total: {}".format(time_total))
            if time_total > 3600 * 24:
                logger.info("time_total > 24 hours, break")
                break

        sampling_successful = lib_sampler.step_active_sampling()

    p_watchdog.terminate()
    p_watchdog.join()
