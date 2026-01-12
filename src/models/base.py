"""
Model Base Class.
"""
from typing import final
from enum import Enum
import yaml


class FeatureSpec(Enum):
    """Enum for different feature specifications of models, and required features for metrics."""

    CONTINUOUS = "continuous"
    EMBEDDING = "embedding"
    TRAJECTORY = "trajectory"
    GENE_EXPRESSION = "gene_expression"
    GRN_INFERENCE = "grn_inference"


class BaseModel:
    def __init__(self, config):
        self.config = config
        self._check_feature_specs()

    @final
    def _check_feature_specs(self):
        """
        Populate the feature specifications required for the metric.
        """
        self.required_feature_specs = None

        # let's use the defined features.yaml to get the features for this model
        with open(self.config.model_features_path, "r") as f:
            features_config = yaml.safe_load(f)

        model_name = self.config.model["name"]

        for model in features_config:
            if model["name"] == model_name:
                self.required_feature_specs = [
                    FeatureSpec(feature) for feature in model["features"]
                ]
                return

        raise ValueError(f"Model features not defined for model: {model_name}")

    def train_and_test(self, dataset):
        """
        Subclasses should implement this method to train and test the model
        on the provided dataset.
        """
        # should be based off of the config's train script and test script
