from hifuku.datagen.batch_sampler import DistributeBatchProblemSampler
from hifuku.datagen.batch_solver import DistributedBatchProblemSolver
from hifuku.domain import ClutteredFridge_SQP
from hifuku.pool import ProblemPool
from hifuku.script_utils import get_project_path

domain = ClutteredFridge_SQP
pp = get_project_path(domain, postfix="metalearn")
task_type = domain.task_type
solver = DistributedBatchProblemSolver(domain.solver_type, domain.solver_config)

init_traj_list = []
n_regression_setting = 10
while len(init_traj_list) < n_regression_setting:
    tasks = [task_type.sample(1) for _ in range(100)]
    resultss = solver.solve_batch(tasks, [None] * 100, use_default_solver=True)
    for results in resultss:
        result = results[0]
        if result.traj is not None:
            init_traj_list.append(result.traj)
    print(f"init_traj_list: {len(init_traj_list)}")


sampler = DistributeBatchProblemSampler()
pool = ProblemPool(domain.task_type, 80)
tasks = sampler.sample_batch(2000, pool)
# solver = DistributedBatchProblemSolver(domain.solver_type, domain.solver_config)
