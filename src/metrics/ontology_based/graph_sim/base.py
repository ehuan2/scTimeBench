"""
Graph Similarity Metric Base Class
"""
from metrics.base import BaseMetric, OutputPathName
from models.base import FeatureSpec
from dataset.filters.lineage import LineageDatasetFilter


class GraphSimMetric(BaseMetric):
    def __init__(self, config, db_manager):
        super().__init__(config, db_manager)

        # ** NOTE: must define the following two attributes **
        self.required_feature_specs = [FeatureSpec.TRAJECTORY]
        self.output_path_name = OutputPathName.GRAPH_SIM
        self.dataset_filters = [LineageDatasetFilter(self.config)]

    def _eval(self, output_path, test_data_path):
        """
        The graph similarity metrics we will be using will take in
        """
        self.graph_pred = None
        self.graph_ref = None

        if self.submetrics:
            for submetric in self.submetrics:
                submetric_instance = submetric(self.config)
                submetric_instance._graph_sim_eval(self.graph_pred, self.graph_ref)
        else:
            self._graph_sim_eval(self.graph_pred, self.graph_ref)

    def _graph_sim_eval(self, graph_pred, graph_ref):
        raise NotImplementedError("Subclasses should implement this method.")
