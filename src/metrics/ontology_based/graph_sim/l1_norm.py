from src.metrics.ontology_based.graph_sim.base import GraphSimMetric


class L1Norm(GraphSimMetric):
    def eval(self, graph_pred):
        """
        The graph similarity metrics we will be using will take in
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def build_reference_objects(self):
        """
        Build the reference graph object needed for graph similarity metrics
        """
