import argparse

from rpbench.interface import PlanningDataset
from rpbench.jaxon.below_table import HumanoidTableReachingTask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=300, help="")
    parser.add_argument("-m", type=int, default=12, help="number of process")

    args = parser.parse_args()
    n_data = args.n
    n_process = args.m

    dataset = PlanningDataset.create(HumanoidTableReachingTask, n_data, n_process)
    dataset.save()
