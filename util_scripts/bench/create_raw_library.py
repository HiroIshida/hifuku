import argparse

from rpbench.interface import PlanningDataset

from hifuku.domain import select_domain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=300, help="")
    parser.add_argument("-m", type=int, default=12, help="number of process")
    parser.add_argument("-domain", type=str, default="humanoid_trr_sqp", help="")
    args = parser.parse_args()

    domain_name: str = args.domain
    domain = select_domain(domain_name)
    task_type = domain.task_type

    args = parser.parse_args()
    n_data = args.n
    n_process = args.m

    dataset = PlanningDataset.create(task_type, n_data, n_process)
    dataset.save()
