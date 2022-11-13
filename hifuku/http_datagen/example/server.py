import numpy as np
from llazy.generation import DataGenerationTask
from skplan.solver.optimization import IKConfig

from hifuku.http_datagen.server import run_server
from hifuku.threedim.tabletop import TabletopIKProblem
from hifuku.types import RawData


class TabletopIKGenerationTask(DataGenerationTask[RawData]):
    def post_init_hook(self) -> None:
        pass

    def generate_single_data(self) -> RawData:
        av_init = np.zeros(10)
        ik_config = IKConfig(disp=False)
        problem = TabletopIKProblem.sample(n_pose=2000)
        results = problem.solve_dummy(av_init, config=ik_config)
        data = RawData.create(problem, results)
        return data


run_server(TabletopIKGenerationTask)
