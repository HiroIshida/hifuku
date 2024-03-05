import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
import tqdm
from mohou.trainer import TrainCache, TrainConfig, train

from hifuku.core import SolutionLibrary
from hifuku.coverage import RealEstAggregate
from hifuku.domain import select_domain
from hifuku.neuralnet import (
    CostPredictor,
    CostPredictorConfig,
    CostPredictorWithEncoder,
    CostPredictorWithEncoderConfig,
    PixelAutoEncoder,
    create_dataset_from_params_and_results,
)
from hifuku.pool import TaskPool
from hifuku.script_utils import create_default_logger, load_compatible_autoencoder


def check_duplication_using_hash(lst):
    hashval_lst = []
    for e in lst:
        h = hashlib.sha256(pickle.dumps(e)).hexdigest()
        hashval_lst.append(h)
    assert len(set(hashval_lst)) == len(lst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", type=str, default="pr2_minifridge_sqp", help="")
    args = parser.parse_args()
    domain_name: str = args.type
    domain = select_domain(domain_name)
    task_type = domain.task_type

    pp = Path("/home/leus/.mohou/hifuku-PR2MiniFridge_SQP-0.1")
    lib = SolutionLibrary.load(pp, torch.device("cuda"))
    traj = lib.init_solutions[-1]
    print("finish determining traj")

    path = Path("./hoge")
    path.mkdir(exist_ok=True)
    n = 100000
    cache_path = path / f"params_results-{n}.pkl"

    if cache_path.exists():
        with cache_path.open("rb") as f:
            params, results = pickle.load(f)
        print("finish loading cache")
    else:
        sampler = domain.get_distributed_batch_sampler()
        pool = TaskPool(task_type)
        params = sampler.sample_batch(n, pool)
        check_duplication_using_hash(params)
        print("finish sampling")

        solver = domain.get_distributed_batch_solver()
        results = solver.solve_batch(params, [traj] * n)

        with cache_path.open("wb") as f:
            pickle.dump((params, results), f)

    success_rate = sum(1 for result in results if result.traj is not None) / n
    print(f"success_rate: {success_rate}")

    ae: PixelAutoEncoder = load_compatible_autoencoder(domain, False, 56)

    dataset = create_dataset_from_params_and_results(
        params, results, domain.solver_config, domain.task_type, None, None
    )
    print("finish creating dataset")

    task = domain.task_type.sample()
    vec = task.export_task_expression(use_matrix=True).get_vector()
    model = CostPredictor(CostPredictorConfig(len(vec), ae.config.dim_bottleneck))

    ae_untrained = PixelAutoEncoder(ae.config)
    model_comb = CostPredictorWithEncoder(CostPredictorWithEncoderConfig(model, ae))

    tconf = TrainConfig()
    tcache = TrainCache.from_model(model_comb)
    logger = create_default_logger(path, "hoge")

    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()
    train(path, tcache, dataset, tconf, early_stopping_patience=10, device=torch.device("cuda"))
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True, show_all=False))

    # 1000 samples ang solve in batch and compare the result
    samples = [domain.task_type.sample() for _ in tqdm.tqdm(range(1000))]
    params = np.array([s.to_task_param() for s in samples])
    solver = domain.get_distributed_batch_solver()
    results = solver.solve_batch(params, [traj] * n)
    print("finish solving")

    # estimate
    reals = []
    ests = []
    for task, result in tqdm.tqdm(zip(samples, results)):
        if result.traj is None:
            reals.append(100)
        else:
            reals.append(result.n_call)
        exp = task.export_task_expression(True)
        mat = exp.get_matrix()
        vec = exp.get_vector()
        mat_tensor = torch.from_numpy(mat).unsqueeze(0).unsqueeze(0).float()
        vec_tensor = torch.from_numpy(vec).unsqueeze(0).float()
        mat_tensor = mat_tensor.to(model_comb.device)
        vec_tensor = vec_tensor.to(model_comb.device)
        with torch.no_grad():
            cost = model_comb((mat_tensor, vec_tensor))
            ests.append(cost.item())
    reals = np.array(reals)
    ests = np.array(ests)
    agg = RealEstAggregate(reals, ests, domain.solver_config.n_max_call)

    logger.info(agg)
    b, coverage = agg.determine_bias(0.1)
    logger.info(f"coverage {coverage}, bias {b}")
