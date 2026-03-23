Method Contribution 🛠️
======================

Overview
--------

This page describes how to add a new method to scTimeBench.

Method contributions usually follow one of two layouts:

1. a method that is installed from PyPI or another Python package index, and
2. a method that vendors a local sub-module under methods/.

Both layouts use the same runner interface, configuration shape, and output contract.

Implementation Layouts
----------------------

PyPI-installed methods
~~~~~~~~~~~~~~~~~~~~~~~

Use this layout when the external method is available as a package and can be
installed during the method setup script. The runner imports the package directly
and focuses on converting scTimeBench inputs into the format the package expects.

Examples in the codebase include:

* `methods/moscot/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/methods/moscot/>`_
* `methods/WOT/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/methods/WOT/>`_
* `methods/PRESCIENT/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/methods/PRESCIENT/>`_
* `methods/MIOFlow/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/methods/MIOFlow/>`_

Vendored sub-module methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this layout when the method source is kept inside the repository or when the
upstream project needs local files and custom requirements. The runner adds the
module directory to sys.path and imports the implementation from the vendored
package.

Examples in the codebase include:

* `methods/MNN/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/methods/MNN/>`_
* `methods/scIMF/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/methods/scIMF/>`_
* `methods/PISDE/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/methods/PISDE/>`_
* `methods/scNODE/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/methods/scNODE/>`_

Runner Structure
----------------

Every method exposes a Python entrypoint under ``methods/<method>/run.py`` and a
companion shell script named train_and_test.sh.

The runner should:

* import main and BaseMethod from scTimeBench.method_utils.method_runner,
* subclass BaseOTMethod when the method is OT-based,
* subclass BaseMethod for all other methods,
* implement train(ann_data, all_tps=None), and
* implement the required generate_* methods for the requested outputs.

OT-specific methods should override get_transport_plan() and may also override
_prepare_generate(). The shared OT base class handles transport-plan caching and
the generation of next-timepoint outputs.

The main entrypoint should follow the existing pattern:

.. code-block:: python

   if __name__ == "__main__":
       main(MyMethodClass)

Output Contract
---------------

Method outputs are controlled by RequiredOutputFiles and are written into the
per-dataset output directory created by the framework.

Common output types include:

* embeddings,
* next-timepoint embeddings,
* next-cell-type predictions, and
* next-timepoint gene expression.

The method class must implement the generators that correspond to the requested
outputs. The training and generation flow is driven by
`src/scTimeBench/method_utils/method_runner.py <https://github.com/li-lab-mcgill/scTimeBench/blob/main/src/scTimeBench/method_utils/method_runner.py>`_.

Setup Script
------------

Each method folder includes a train_and_test.sh script that prepares dependencies
and launches the Python runner.

The setup script usually does the following:

* installs the method dependency from PyPI or from a local requirements file,
* installs the scTimeBench package in editable mode,
* validates the YAML config path, and
* runs python ./methods/<method>/run.py --yaml_config <config>.

For vendored methods, the script often installs a local requirements.txt from the
sub-module directory before calling the runner.

Configuration
-------------

The method runner receives a YAML file with the following important fields:

* dataset_pkl_path: pickled AnnData dataset produced by the benchmark pipeline,
* output_path: where the method should write its outputs,
* required_outputs: the list of files that must be produced, and
* method.metadata: method-specific hyperparameters.

The configuration is prepared by the shared method runner and passed to the method
class as self.config.

Testing
-------

Run the method through its train_and_test.sh script and verify that the expected
output files are produced. If the method uses caching, confirm that repeated runs
reuse the cached artifacts when appropriate.

Checklist
---------

* runner added under methods/<method>/run.py
* train_and_test.sh added or updated
* method class inherits from BaseMethod or BaseOTMethod
* required generate_* methods implemented
* dependencies documented in the setup script
* the expected RequiredOutputFiles are produced
