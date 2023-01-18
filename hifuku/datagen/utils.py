import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def split_number(num, div):
    return [num // div + (1 if x < num % div else 0) for x in range(div)]


def split_indices(n_problem_total: int, n_problem_list: List[int]) -> List[List[int]]:
    indices = np.array(list(range(n_problem_total)))
    indices_list = []
    head = 0
    for n_problem in n_problem_list:
        tail = head + n_problem
        indices_list.append(indices[head:tail].tolist())
        head = tail
    assert sum([len(ii) for ii in indices_list]) == n_problem_total
    return indices_list
