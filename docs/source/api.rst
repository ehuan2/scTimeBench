API Reference 📒
================

Core Runtime
------------

These modules control configuration parsing, persistence, and the top-level
benchmark entrypoint.

.. automodule:: scTimeBench.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: scTimeBench.database
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: scTimeBench.main
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Infrastructure
----------------------

These modules define dataset loading, preprocessing, shared constants, and
utility helpers used throughout the benchmark.

.. automodule:: scTimeBench.shared.constants
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: scTimeBench.shared.dataset.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: scTimeBench.shared.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: scTimeBench.shared.helpers
   :members:
   :undoc-members:
   :show-inheritance:

Method Execution
----------------

These modules provide the method runner interface and the helper used by the
benchmark to launch methods and collect their outputs.

.. automodule:: scTimeBench.method_utils.method_runner
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: scTimeBench.method_utils.ot_method_runner
   :members:
   :undoc-members:
   :show-inheritance:

Metric Framework
----------------

These modules define the metric base class and the method manager used to bind
datasets to method outputs during evaluation.

.. automodule:: scTimeBench.metrics.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: scTimeBench.metrics.method_manager
   :members:
   :undoc-members:
   :show-inheritance:

Trajectory Inference
--------------------

These modules implement the trajectory inference abstractions and concrete
inference strategies used by the metrics.

.. automodule:: scTimeBench.trajectory_infer.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: scTimeBench.trajectory_infer.classifier
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: scTimeBench.trajectory_infer.kNN
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: scTimeBench.trajectory_infer.ot
   :members:
   :undoc-members:
   :show-inheritance:
