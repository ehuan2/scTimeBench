from crispy_fishstick.metrics.ontology_based.graph_sim.base import AdjacencyMatrixType
from crispy_fishstick.metrics.ontology_based.graph_sim.ged import GraphEditDistance
from crispy_fishstick.metrics.ontology_based.graph_sim.jaccard_similarity import (
    JaccardSimilarity,
)
import numpy as np


def test_graph_edit_distance():
    """
    Tests that the Graph Edit Distance metric gives expected results.
    """
    ged = GraphEditDistance(None, None, {})

    def wrapper_graph_sim_eval(adj_pred, adj_ref):
        return ged._graph_sim_eval(
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_pred,
            },
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_ref,
            },
        )

    assert wrapper_graph_sim_eval(
        np.array(
            [
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
            ]
        ),
    ) == np.sqrt(2), "Graph Edit Distance calculation is incorrect."

    assert (
        wrapper_graph_sim_eval(
            np.array(
                [
                    [0, 1],
                    [0, 0],
                ]
            ),
            np.array(
                [
                    [0, 1],
                    [0, 0],
                ]
            ),
        )
        == 0
    ), "Graph Edit Distance for identical graphs should be zero."

    assert (
        wrapper_graph_sim_eval(
            np.array(
                [
                    [0, 1],
                    [0, 0],
                ]
            ),
            np.array(
                [
                    [0, 1],
                    [1, 0],
                ]
            ),
        )
        == 1
    ), "Graph Edit Distance is incorrect for single edge difference."


def test_jaccard_similarity():
    """
    Tests that the Jaccard Similarity metric gives expected results.
    """
    js = JaccardSimilarity(None, None, {})

    def wrapper_graph_sim_eval(adj_pred, adj_ref):
        return js._graph_sim_eval(
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_pred,
            },
            {
                AdjacencyMatrixType.UNWEIGHTED: adj_ref,
            },
        )

    # Test identical graphs
    assert (
        wrapper_graph_sim_eval(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                ]
            ),
        )
        == 1.0
    ), "Jaccard Similarity for identical graphs should be 1.0."

    # Test completely different graphs
    assert (
        wrapper_graph_sim_eval(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                ]
            ),
        )
        == 0.0
    ), "Jaccard Similarity for completely different graphs should be 0.0."

    # Test partial overlap
    assert (
        wrapper_graph_sim_eval(
            np.array(
                [
                    [0, 1],
                    [0, 0],
                ]
            ),
            np.array(
                [
                    [0, 1],
                    [1, 0],
                ]
            ),
        )
        == 0.5
    ), "Jaccard Similarity with 1 common edge and 2 total edges should be 0.5."
