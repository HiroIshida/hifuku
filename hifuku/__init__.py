import importlib.metadata

import packaging.version
from ompl import turn_off_logger


def is_plainmp_old():
    version_str = importlib.metadata.version("plainmp")
    return packaging.version.parse(version_str) < packaging.version.parse("0.0.8")


# turn of ompl-thin packages logger
turn_off_logger()

# turn off plainmp logger (in-house build version of ompl inside plainmp)
if not is_plainmp_old():
    from plainmp.ompl_solver import set_log_level_none

    set_log_level_none()
