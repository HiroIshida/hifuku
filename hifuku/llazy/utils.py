from typing import List


def split_number(num: int, div: int) -> List[int]:
    return [num // div + (1 if x < num % div else 0) for x in range(div)]
