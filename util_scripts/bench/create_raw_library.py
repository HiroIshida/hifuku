import argparse

from rpbench.interface import PlanningDataset
from rpbench.jaxon.below_table import HumanoidTableReachingTask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-domain", type=str, help="")
    parser.add_argument("-n", type=int, default=300, help="")
    parser.add_argument("-m", type=int, default=2, help="number of process")

    dataset = PlanningDataset.create(HumanoidTableReachingTask, 10000, 12)
    dataset.save()
