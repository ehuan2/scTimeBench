Default Run 🚦
==============
TODO : Update documentation for sqlite database management and csv extraction.

Overview
--------

This page describes the default scTimeBench benchmark flow.

In the standard setup, you choose a method in the YAML file, list the compatible
metrics, and let the benchmark resolve the default datasets for those metrics.
The dataset definitions, preprocessing steps, and metric-group defaults are then
pulled from the shared dataset registry.

The datasets, configuration files and scripts for methods implemented in the scTimeBench paper can be downloaded from `Zenodo <https://doi.org/10.5281/zenodo.19196641>`_.

Start from a config file
------------------------

The benchmark is started from the command line with a YAML file:

.. code-block:: bash

   scTimeBench --config configs/scNODE/gex.yaml

The main entrypoint is implemented in `src/scTimeBench/main.py <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/main.py>`_.
Configuration parsing and validation live in `src/scTimeBench/config.py <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/config.py>`_.

Select the method
-----------------

The method section tells scTimeBench which runner to use and where to find the
method-specific shell script. A minimal example looks like this:

.. code-block:: yaml

   method:
     name: scNODE
     train_and_test_script: ./methods/scNODE/train_and_test.sh

The method name must match the registered method class, and the script must point
to the correct train_and_test.sh file under methods/.

Select compatible metrics
-------------------------

The metrics list controls which evaluation families run for the chosen method.
Each metric class declares the dataset class names it supports, so the config only
needs to name the metric classes.

Example:

.. code-block:: yaml

   metrics:
     - name: GraphSimMetric

If multiple metrics are listed, they are evaluated in order. Some configs repeat
a metric with different parameters, such as alternate trajectory inference models.

Let the metric choose default datasets
--------------------------------------

If datasets are not listed explicitly in the config, scTimeBench uses the default
datasets for the metric group. Those defaults are defined in
`src/scTimeBench/shared/dataset/default_datasets.yaml <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/shared/dataset/default_datasets.yaml>`_.

For example, the embedding and ontology-based metric groups use the default tags
defined in that file, and then the framework resolves those tags to full dataset
definitions before preprocessing starts.

Dataset overrides & customization
---------------------------------

You can supply a datasets section to override the defaults. This is useful when
you want a smaller subset, a different default tag, or a custom dataset entry.

Tag-based usage is the most common form:

.. code-block:: yaml

   datasets:
     - tag: defaultGarciaAlonso
     - tag: defaultGarciaAlonsoPseudotimeEvenCells

You can also provide a full dataset definition with a path and preprocessing steps
when the dataset is not already in the shared registry.

Custom preprocessing
--------------------

Dataset preprocessing is configured per dataset entry and is executed in order.
Typical preprocessing steps include:

* lineage filtering,
* pseudotime inference,
* timepoint rounding,
* log-normalization, and
* train/test splitting.

Example:

.. code-block:: yaml

   datasets:
     - name: GarciaAlonsoDataset
       data_path: ./data/garcia-alonso/human_germ.h5ad
       data_preprocessing_steps:
         - name: LineageDatasetFilter
           cell_lineage_file: ./cell_lineages/germ/cell_line.txt
           cell_equivalence_file: ./cell_lineages/germ/equal_names.txt
         - name: RoundCellsToTimepoint
           min_cells_per_timepoint: 10
         - name: LogNormPreprocessor
         - name: CopyTrainTest

The config parser accepts either a full dataset definition or a tag-only entry,
but not a mix of tag and explicit dataset fields.

Understand run modes
--------------------

The --run_type option controls how much of the pipeline executes:

* auto_train_test: train the method and then evaluate the metrics,
* preprocess: only preprocess and prepare outputs,
* eval_only: only evaluate previously generated outputs, and
* train_only: train the method but skip metric evaluation.

The default run mode is auto_train_test.

Validate paths and outputs
--------------------------

The config loader checks that dataset paths, lineage files, and method scripts
exist before the run starts. It also requires:

* method.name,
* metrics, and
* train_and_test_script when auto_train_test is used.

The benchmark then creates per-method output directories, caches processed data,
and verifies the required output files before metric evaluation.

Checklist
---------

* config file points to the desired method runner
* metrics listed are compatible with that method's outputs
* dataset tags match entries in the shared dataset registry
* preprocessing steps are ordered correctly
* paths exist for data files, lineage files, and method scripts
