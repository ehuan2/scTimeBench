"""
Ma et al. (2023) dataset.
"""

from crispy_fishstick.shared.dataset.base import BaseDataset, ObservationColumns
import scanpy as sc


class MaDataset(BaseDataset):
    def _load_data(self):
        """
        Load the Ma et al. dataset.
        """
        print("Loading Ma et al. dataset...")
        # read from the dataset data_path
        data_path = self.dataset_dict["data_path"]
        self.data = sc.read_h5ad(data_path)

        # now let's filter out all the datapoints that are low quality
        # i.e. nan age and nan cell type
        # rename these columns to standard names
        cell_type_col = "cell_type"
        timepoint_col = "timepoint"
        cell_type_candidates = [
            "celltype",
            "cell_type",
            "Celltype",
            "CellType",
        ]
        timepoint_candidates = [
            "PCW",
            "GestationAge",
            "gestation_age",
            "timepoint",
        ]

        for candidate in cell_type_candidates:
            if candidate in self.data.obs.columns:
                cell_type_col = candidate
                break

        for candidate in timepoint_candidates:
            if candidate in self.data.obs.columns:
                timepoint_col = candidate
                break

        if cell_type_col is None or timepoint_col is None:
            missing = []
            if cell_type_col is None:
                missing.append("cell type column")
            if timepoint_col is None:
                missing.append("timepoint column")
            available = ", ".join(self.data.obs.columns)
            raise ValueError(
                f"MaDataset missing {', '.join(missing)}; available columns: {available}"
            )

        self.data.obs = self.data.obs.rename(
            columns={
                cell_type_col: ObservationColumns.CELL_TYPE.value,
                timepoint_col: ObservationColumns.TIMEPOINT.value,
            }
        )
        print("Ma et al. dataset loaded successfully.")

        print(
            f"Cell types: {self.data.obs[ObservationColumns.CELL_TYPE.value].unique()}"
        )
