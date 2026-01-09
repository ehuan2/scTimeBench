"""
Base class for all metrics. They should all implement the eval method, and
depend on the dataset that they belong to.
"""
from src.config import Config

# also store a registry of metrics of name to class
METRIC_REGISTRY = {}


class BaseMetric:
    def __init__(self, name: str, dataset: str, config: Config):
        self.name = name
        self.dataset = dataset
        self.config = config
        self.build_reference_objects()
        METRIC_REGISTRY[name] = self.__class__

    def eval(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def build_reference_objects(self):
        """
        Build any reference objects needed for the metric.

        If the dataset is known, we can load precomputed reference objects here.
        """
        raise NotImplementedError("Subclasses should implement this method.")
