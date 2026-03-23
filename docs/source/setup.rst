Installation & Setup
====================

This page covers the recommended environment setup for running the benchmark,
developing methods, and building the documentation.

Install: pip
~~~~~~~~~~~~

If the external dependencies such as pypsupertime or sceptic are not needed, you
can install the benchmark with pip as follows:

.. code-block:: bash

   pip install -e ".[benchmark]"

This installs the benchmark dependencies used for the default pipeline. For
method development, install without the extra benchmarking requirements with:

.. code-block:: bash

   pip install -e .

Additional optional dependency groups are defined in the `pyproject.toml <https://github.com/li-lab-mcgill/scTimeBench/blob/main/pyproject.toml>`_ file.

Install: UV
~~~~~~~~~~~

The repository also supports uv, which is useful for managing the benchmark and
method environments. Install uv and then run:

.. code-block:: bash

   uv sync

This creates the default environment for running the benchmark code and the
shared runners inside scTimeBench.

If you only need a lighter environment, for example for a specific method, you
can use:

.. code-block:: bash

   uv sync --no-dev

For local development or testing, install one or more extra groups as needed:

.. code-block:: bash

   uv sync --extra test --extra dev --extra benchmark

or:

.. code-block:: bash

   uv sync --all-extras

Python Version
~~~~~~~~~~~~~~

The project targets Python 3.10. Use:

.. code-block:: bash

   uv python install 3.10
   uv python pin 3.10

before running uv sync.

Repository Layout
~~~~~~~~~~~~~~~~~

* `configs/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/configs/>`_ contains example benchmark YAML files.
* `docs/source/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/docs/source/>`_ contains the documentation source files.
* `extern/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/extern/>`_ contains vendored external code used by some methods.
* `methods/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/methods/>`_ contains method wrappers and their setup scripts.
* `src/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/src/>`_ contains the scTimeBench package itself.
* `test/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/test/>`_ contains the benchmark's test suite.

Example Run
-----------

Run the benchmark with a configuration file such as `configs/scNODE/gex.yaml <https://github.com/li-lab-mcgill/scTimeBench/blob/main/configs/scNODE/gex.yaml>`_:

.. code-block:: bash

   scTimeBench --config configs/scNODE/gex.yaml

You can also run the package entrypoint directly:

.. code-block:: bash

   python src/scTimeBench/main.py --config configs/scNODE/gex.yaml

Contributing
------------

If you want to contribute, install the development and benchmark dependencies
with one of the following:

.. code-block:: bash

   pip install -e ".[dev, benchmark]"

or:

.. code-block:: bash

   uv sync --extra dev --extra benchmark

To enable the autoformatter and pre-commit hooks, run:

.. code-block:: bash

   pre-commit install

Testing
-------

Run the benchmark test suite with:

.. code-block:: bash

   pytest test

from the repository root, or simply run ``pytest`` from inside `test/ <https://github.com/li-lab-mcgill/scTimeBench/tree/main/test/>`_.

See the `pytest documentation <https://docs.pytest.org/en/stable/>`_ for more
information. A useful flag is -s to view full output.
