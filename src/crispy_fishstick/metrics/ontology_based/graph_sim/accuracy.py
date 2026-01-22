from crispy_fishstick.metrics.ontology_based.graph_sim.base import (
    GraphSimMetric,
    AdjacencyMatrixType,
)

import numpy as np
import logging


class GraphAccuracy(GraphSimMetric):
    def _graph_sim_eval(self, graph_pred, graph_ref):
        """
        The graph similarity metrics we will be using will take in
        """
        graph_pred_adj = graph_pred[AdjacencyMatrixType.UNWEIGHTED]
        graph_ref_adj = graph_ref[AdjacencyMatrixType.UNWEIGHTED]

        # calculate the accuracy as the number of correct edges over total edges
        correct_edges = np.sum(graph_pred_adj == graph_ref_adj)
        total_edges = graph_ref_adj.shape[0] * graph_ref_adj.shape[1]
        accuracy = correct_edges / total_edges

        logging.debug(f"Graph Accuracy: {accuracy}")
        return accuracy
