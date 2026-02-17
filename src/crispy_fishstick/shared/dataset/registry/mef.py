"""
MEF dataset.
"""

from crispy_fishstick.shared.dataset.base import BaseDataset, ObservationColumns
import scanpy as sc


class MEFDataset(BaseDataset):
    def _load_data(self):
        """
        Load the MEF dataset.
        """
        print("Loading MEF dataset...")
        # read from the dataset data_path
        data_path = self.dataset_dict["data_path"]
        self.data = sc.read_h5ad(data_path)

        self.data.obs = self.data.obs.rename(
            columns={
                "timepoint": ObservationColumns.TIMEPOINT.value,
            }
        )
        print("MEF dataset loaded successfully.")

        print(
            f"Timepoints: {self.data.obs[ObservationColumns.TIMEPOINT.value].unique()}"
        )
        print("No celltypes available for this dataset as of yet.")
