"""
Base class for all metrics. They should all implement the eval method, and
depend on the dataset that they belong to.
"""
from config import Config
from typing import final
from models.base import MODEL_REGISTRY
from dataset.base import DATASET_REGISTRY

METRIC_REGISTRY = {}


def register_metric(cls):
    """Decorator to register a metric class in the METRIC_REGISTRY."""
    METRIC_REGISTRY[cls.__name__] = cls


# also store a registry of metrics of name to class
class BaseMetric:
    def __init__(self, config: Config):
        self.config = config
        # 1) check the required feature specs match up
        self._check_feature_specs()
        self._init_model()
        self._check_model_feature_specs()

    def __init_subclass__(cls):
        """
        Automatically register subclasses in the METRIC_REGISTRY.

        This allows for hierarchical structuring of metrics, where
        subclasses are tracked under their parent classes.
        """
        register_metric(cls)

        # we want to clear the submetrics list for each subclass
        cls.submetrics = []

        for base in cls.__bases__:
            if hasattr(base, "submetrics"):
                base.submetrics.append(cls)

    # ** METRIC EVALUATION SECTION, main function to be called (no others should be called) **
    @final
    def eval(self, *args, **kwargs):
        """
        Evaluates the model on the dataset according to the metric.

        First, however, this method will:
        1) preprocess the dataset according to the metric's needs
        2) train the model on the preprocess dataset
        3) grab whatever the model needs to evaluate the metric
        4) evaluate the metric
        """
        # 1) initialize the dataset splits dependent on the metric
        self._init_dataset()

        # TODO: figure this out below
        # # 2) train the model on the dataset
        # self.model.train(self.dataset)

        # # 3) evaluate the model according to the metric
        # self.model.test(self.dataset)

        # 4) evaluate the metric based on the model outputs
        self._eval()

    # ** METRIC FEATURE SPECS SECTION **
    @final
    def _check_feature_specs(self):
        """
        Populate the feature specifications required for the metric.
        """
        self.required_feature_specs = None
        self._populate_feature_specs()
        assert (
            self.required_feature_specs is not None
        ), "Subclasses must define required_feature_specs"

    def _populate_feature_specs(self):
        """
        Subclasses should implement this method to define
        their required feature specifications.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @final
    def _init_model(self):
        """
        Initialize the model based on the configuration.
        """
        model_name = self.config.model["name"]
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model {model_name} not found in registry.")
        model_class = MODEL_REGISTRY[model_name]
        self.model = model_class(self.config)

    @final
    def _check_model_feature_specs(self):
        """
        Check if the model's feature specifications satisfy the metric's requirements.
        """
        model_feature_specs = self.model.required_feature_specs
        required_specs = self.required_feature_specs

        for spec in required_specs:
            if spec not in model_feature_specs:
                raise ValueError(
                    f"Model {self.config.model['name']} does not satisfy the required feature spec: {spec}"
                )

    # ** PREPROCESSING DATASET SECTION **
    def _preprocess_dataset(self):
        """
        Subclasses should implement this method to define
        their required dataset splits.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _populate_dataset_filters(self):
        """
        Populate the dataset filters required for the metric.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @final
    def _init_dataset(self):
        """
        Initializes the dataset by first checking if the processed dataset is
        already cached, and if not, to go through the processing steps.

        Steps:
        1. Calls _populate_dataset_filters to grab the necessary dataset filters.
        2. Checks if the cache exists based on the dataset filters and dataset.
        3. Applies each filter to the dataset in sequence.
        4. Caches the processed dataset for future use.
        """
        self._populate_dataset_filters()

        # TODO
        # now we check the cache based on the set preprocessed directory
        # stop if the cache exists

        # if the cache does not exist, we go ahead and process the dataset
        self.dataset = DATASET_REGISTRY[self.config.dataset["name"]](self.config)
        self.dataset.load_data()
        self.dataset.apply_filters(self.dataset_filters)

        # TODO
        # now we cache the processed dataset for future use
