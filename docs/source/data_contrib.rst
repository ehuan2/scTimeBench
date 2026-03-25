Dataset Contribution 🗂️
=======================

Overview
--------

This page describes how to add a new dataset to scTimeBench.

Dataset contributions usually require three parts:

1. a dataset loader in the dataset registry,
2. a default dataset configuration, and
3. inclusion in the supported metric groups.

Steps for Contribution
----------------------

1. Format the dataset
~~~~~~~~~~~~~~~~~~~~~
   Ensure the source data can be loaded into an AnnData object and that the following
   observation columns are available after loading:

   * cell type, mapped to ObservationColumns.CELL_TYPE
   * timepoint, mapped to ObservationColumns.TIMEPOINT

   If a dataset does not contain one of these fields, follow the existing dataset
   loaders and provide a sensible placeholder or derived value.

2. Add a dataset loader
~~~~~~~~~~~~~~~~~~~~~~~
   Create a loader in `src/scTimeBench/shared/dataset/registry/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/src/scTimeBench/shared/dataset/registry/>`_
   and follow the existing naming convention: the class name should end in Dataset,
   and the module name should be snake case.

   A minimal loader looks like this:

   .. code-block:: python

      """
      Dataset name.
      """

      import scanpy as sc

      from scTimeBench.shared.dataset.base import BaseDataset, ObservationColumns


      class ExampleDataset(BaseDataset):
          def _load_data(self):
              """Load the dataset into self.data."""
              data_path = self.dataset_dict["data_path"]
              self.data = sc.read_h5ad(data_path)

              self.data.obs = self.data.obs.rename(
                  columns={
                      "cell_type": ObservationColumns.CELL_TYPE.value,
                      "timepoint": ObservationColumns.TIMEPOINT.value,
                  }
              )

   Existing loaders in `src/scTimeBench/shared/dataset/registry/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/src/scTimeBench/shared/dataset/registry/>`_
   show the expected patterns for datasets with and without explicit cell-type labels.

   If your dataset has no cell-type labels, set all cells to ``unknown``:

   .. code-block:: python

      self.data.obs[ObservationColumns.CELL_TYPE.value] = "unknown"

   If your dataset does not have timepoint labels, generate pseudotime labels as
   an alternative.
   
   Remember to export the new class from
   `src/scTimeBench/shared/dataset/registry/__init__.py <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/shared/dataset/registry/__init__.py>`_.

3. Add a default dataset entry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Register the dataset in `src/scTimeBench/shared/dataset/default_datasets.yaml <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/shared/dataset/default_datasets.yaml>`_
   so that it can be loaded through a dataset tag.

   The preprocessing steps are executed in order. Typical steps include lineage
   filtering, pseudotime inference, timepoint rounding, log-normalization, and
   the final train/test split.

   .. code-block:: yaml

      datasets:
        - name: GarciaAlonsoDataset
          tag: defaultGarciaAlonso
          data_path: ./data/garcia-alonso/human_germ.h5ad
          data_preprocessing_steps:
            - name: LineageDatasetFilter
              cell_lineage_file: ./cell_lineages/germ/cell_line.txt
              cell_equivalence_file: ./cell_lineages/germ/equal_names.txt
            - name: RoundCellsToTimepoint
              min_cells_per_timepoint: 10
            - name: LogNormPreprocessor
            - name: CopyTrainTest

   If the dataset is used only for optional ontology-based workflows, add a matching
   entry in `src/scTimeBench/shared/dataset/optional_datasets.yaml <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/shared/dataset/optional_datasets.yaml>`_.

4. Add the dataset to supported metric groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Update the metric defaults so the new tag is discoverable by the relevant metric
   families. The current metric groups are defined in:

   * `src/scTimeBench/metrics/embeddings/base.py <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/metrics/embeddings/base.py>`_
   * `src/scTimeBench/metrics/ontology_based/base.py <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/metrics/ontology_based/base.py>`_
   * `src/scTimeBench/metrics/gex_prediction/base.py <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/metrics/gex_prediction/base.py>`_

   If the new dataset belongs to a group, add its tag to the matching dataset list in
   `src/scTimeBench/shared/dataset/default_datasets.yaml <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/shared/dataset/default_datasets.yaml>`_
   and ensure the metric subclass supports the dataset class name.

5. Upload the data
~~~~~~~~~~~~~~~~~~
   Upload the dataset to a file hosting service such as Google Drive, Zenodo or Kaggle. This will facilitate our ability to update our Zenodo data release with your contributions.

6. Open a pull request
~~~~~~~~~~~~~~~~~~~~~~
   After the loader, configuration, and data references are in place, open a pull
   request with a clear description of:

   * the dataset source,
   * the preprocessing applied,
   * any caveats or missing annotations, and
   * the intended use cases.


Checklist
---------

* dataset loader added under ``src/scTimeBench/shared/dataset/registry/``
* dataset exported from ``registry/__init__.py``
* default dataset tag added to the appropriate YAML file
* metric group defaults updated, if needed
* data uploaded and linked in the pull request