"""
Definitions to be shared by the benchmark and the model implementations.
"""
from enum import Enum


class ObservationColumns(Enum):
    CELL_TYPE = "cell_type"
    TIMEPOINT = "timepoint"


# TODO: add in required constraints for what the model outputs
# TODO: should look like per metric
