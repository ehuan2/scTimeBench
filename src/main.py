"""
main.py. Entrypoint for measuring trajectories in single-cell data,
particularly involving gene regulatory networks and cell lineage information.
"""
from config import Config

if __name__ == "__main__":
    config = Config()

    # now based on this config, we want to get:
    # 1) the model that we'll be using
    # 2) the dataset that we'll be using
    # 3) the metrics that we'll be using
