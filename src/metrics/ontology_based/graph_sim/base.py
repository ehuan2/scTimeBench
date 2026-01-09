"""
Graph Similarity Metric Base Class
"""
from metrics.base import BaseMetric


class GraphSimMetric(BaseMetric):
    def eval(self, graph_pred):
        """
        The graph similarity metrics we will be using will take in
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def build_reference_objects(self):
        """
        Build the reference graph object needed for graph similarity metrics
        """
