"""
Base filter for datasets. Every metric will likely require different splits of the
data, so this base class will define the necessary interface for dataset preprocessing.
"""

from enum import Enum

# ** DATASET FILTERING SECTION **
DATASET_FILTER_REGISTRY = {}


def register_dataset_filter(cls):
    """Decorator to register a dataset filter class in the DATASET_FILTER_REGISTRY."""
    DATASET_FILTER_REGISTRY[cls.__name__] = cls


class BaseDatasetFilter:
    def __init__(self, config):
        self.config = config

    def __init_subclass__(cls):
        """
        Automatically register subclasses in the DATASET_FILTER_REGISTRY.
        """
        register_dataset_filter(cls)

    def filter(self, dataset):
        """
        Subclasses should implement this method to filter and split the dataset
        according to the metric's requirements.
        """
        raise NotImplementedError("Subclasses should implement this method.")


# ** DATASET INFORMATION **
DATASET_REGISTRY = {}


def register_dataset(cls):
    """Decorator to register a dataset class in the DATASET_REGISTRY."""
    DATASET_REGISTRY[cls.__name__] = cls


class ObservationColumns(Enum):
    CELL_TYPE = "cell_type"
    TIMEPOINT = "timepoint"


class BaseDataset:
    def __init__(self, config):
        self.config = config

    def __init_subclass__(cls):
        register_dataset(cls)

    def _load_data(self):
        """
        Subclasses should implement this method to load the dataset.
        Each dataset might require its own loading mechanism, as well as preprocessing
        mechanisms, but the BaseDatasetFilter should hopefully work on all datasets.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def load_data(self):
        """
        This ensures that the dataset loading is done properly.

        We require the following:
        1. Load the data from the source.
        2. Include observation metadata of cell_type, and timepoint.
        3. Drop everything else not required, to speed up processing.
        """
        self._load_data()

        # now let's verify that the necessary columns exist
        assert hasattr(
            self, "data"
        ), "Dataset must have a 'data' attribute after loading."
        assert (
            ObservationColumns.CELL_TYPE.value in self.data.obs.columns
        ), f"Dataset must have '{ObservationColumns.CELL_TYPE.value}' in observation metadata."
        assert (
            ObservationColumns.TIMEPOINT.value in self.data.obs.columns
        ), f"Dataset must have '{ObservationColumns.TIMEPOINT.value}' in observation metadata."

    def apply_filters(self, filters):
        """
        Apply a list of dataset filters to the dataset.
        """
        for dataset_filter in filters:
            self = dataset_filter.filter(self)
        return self
