from rpbench.interface import PlanningDataset
from rpbench.jaxon.below_table import HumanoidTableReachingTask

dataset = PlanningDataset.create(HumanoidTableReachingTask, 10000, 12)
dataset.save()
