"""
main.py. Entrypoint for measuring trajectories in single-cell data,
particularly involving gene regulatory networks and cell lineage information.
"""
from config import METRIC_REGISTRY

# required to register metrics
import metrics

if False:
    metrics  # to avoid unused import warning

if __name__ == "__main__":
    # config = Config()

    print(METRIC_REGISTRY)  # For debugging: print registered metrics
