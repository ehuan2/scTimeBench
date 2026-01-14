"""
kNN implementation for trajectory inference.
"""
from trajectory_infer.base import BaseTrajectoryInferMethod


class kNN(BaseTrajectoryInferMethod):
    def __init__(self, traj_config):
        super().__init__(traj_config)
        # sets the default number of neighbors
        self.n_neighbors = traj_config.get("n_neighbors", 5)

    def _method_infer_trajectory(self, ann_data):
        """
        Infer the trajectory using kNN graph-based method.

        1. We can accomplish this by first separating each embedding based on time.
        2. Then, for each time point, we find the k nearest neighbors in the next time point's
        embedding space.
        3. Finally, we consolidate the cell types per time point based on the kNN results.
        """
        return None
