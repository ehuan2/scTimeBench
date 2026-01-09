"""
Suo et al. (2022) dataset.
"""

from dataset.base import BaseDataset, ObservationColumns
import scanpy as sc


class SuoDataset(BaseDataset):
    def _load_data(self):
        """
        Load the Suo et al. dataset.
        """
        print("Loading Suo et al. dataset...")
        # read from the config data_path
        data_path = self.config.dataset["data_path"]
        self.data = sc.read_h5ad(data_path)

        # now let's filter out all the datapoints that are low quality
        # i.e. nan age and nan cell type

        # then, let's only keep the necessary columns in obs
        obs_columns_to_keep = ["celltype_annotation", "age"]

        # and then rename these columns to standard names
        self.data.obs = self.data.obs[obs_columns_to_keep]
        self.data.obs = self.data.obs.rename(
            columns={
                "celltype_annotation": ObservationColumns.CELL_TYPE.value,
                "age": ObservationColumns.TIMEPOINT.value,
            }
        )

        # let's print the var table to see what we can drop
        var_columns_to_keep = []  # keep all for now
        if var_columns_to_keep:
            self.data.var = self.data.var[var_columns_to_keep]

        print("Dataset loaded with the following observation columns:")
        print(self.data.obs.columns.tolist())

        print("Suo et al. dataset loaded successfully.")
