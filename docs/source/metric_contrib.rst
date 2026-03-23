Metric Contribution 🎯
======================

Overview
--------

This page describes how to add a new metric to scTimeBench.

Metric implementation is centered around the class hierarchy in
`src/scTimeBench/metrics/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/src/scTimeBench/metrics/>`_.

Metric Base Classes
-------------------

All metrics inherit from BaseMetric, which provides the shared evaluation flow,
database logging, dataset loading, and method-output validation.

Common family base classes include:

* `EmbeddingMetrics <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/metrics/embeddings/base.py>`_
* `OntologyBasedMetrics <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/metrics/ontology_based/base.py>`_
* `GexPredictionMetrics <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/metrics/gex_prediction/base.py>`_
* `TrajectoryEmbeddingMetrics <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/metrics/embeddings/trajectory/base.py>`_

Use these family classes when the new metric fits an existing output type. Add a
new family only when the evaluation flow is materially different.

Metric Hierarchy
----------------

Metric classes are registered automatically through the BaseMetric subclass
mechanism. A metric can be a direct leaf class or a parent class that exposes
submetrics.

Guidelines:

* implement _defaults() to declare configurable parameters and their defaults,
* implement _setup_supported_datasets() to declare the supported dataset class names,
* implement _setup_method_output_requirements() to declare the required method outputs,
* implement _prep_kwargs_for_submetric_eval() to pass the right arguments into the evaluator, and
* implement _submetric_eval() or the family-specific evaluator such as _embedding_eval().

Helper or abstract classes that are not meant to run directly should use the
skip_metric decorator where appropriate.

Dataset Matching
----------------

Metrics do not match on dataset tags. They match on dataset class names, for
example MaDataset, SuoDataset, or GarciaAlonsoDataset.

The default dataset tags used by the benchmark are configured separately in:

* `src/scTimeBench/shared/dataset/default_datasets.yaml <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/shared/dataset/default_datasets.yaml>`_
* `src/scTimeBench/shared/dataset/optional_datasets.yaml <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/shared/dataset/optional_datasets.yaml>`_

When adding a new dataset, update both the dataset registry and the metric group
lists so the dataset is discoverable by the right metrics.

Metric Groups
-------------

The framework uses metric groups to choose default datasets when a run does not
specify them explicitly.

Current groups are defined in the shared dataset YAML and mapped to metric classes
such as:

* embedding metrics,
* ontology-based metrics, and
* gene-expression prediction metrics.

If your metric belongs to a new family, define a new default_dataset_group and
add a matching entry in the YAML file.

Required Outputs
----------------

The metric should request the outputs that it needs from the method runner. The
required files are defined with RequiredOutputFiles and are checked before
evaluation begins.

Examples include:

* EMBEDDING (embeddings for observed cells),
* NEXT_TIMEPOINT_EMBEDDING (embeddings for projected cells),
* NEXT_CELLTYPE (predicted cell types of projected cells), and
* NEXT_GEX (predicted gene expression of projected cells).

For metrics with multiple acceptable output combinations, use a list of lists in
required_outputs.

Validation
----------

Confirm that the metric runs through the full pipeline:

* dataset selection,
* preprocessing,
* method execution,
* required-output validation, and
* evaluation/database logging.

If the metric is a submetric family, verify that each leaf metric is discovered
and evaluated in the expected order.

Checklist
---------

* metric class added under `src/scTimeBench/metrics/ <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/metrics/>`_
* family base class chosen correctly
* supported datasets listed by class name
* default dataset group set when needed
* required method outputs declared
* evaluation logic covered by tests
