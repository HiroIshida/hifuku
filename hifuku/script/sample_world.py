import time

import numpy as np

from hifuku.tabletop import TabletopIKProblem

# np.random.seed(1)


if __name__ == "__main__":
    while True:
        print("trying...")
        problem = TabletopIKProblem.sample()
        av_init = np.zeros(10)
        res = problem.solve(av_init)
        if res.success:
            break
    problem.visualize(res.x)
    print(res.success)
    time.sleep(10)
