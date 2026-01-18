"""
Suo et al. (2022) dataset.
"""

from crispy_fishstick.shared.dataset.base import BaseDataset


class DummyDataset(BaseDataset):
    def _load_data(self):
        """
        Fake load data
        """
        print("Dummy dataset meant for testing... Loading...")
