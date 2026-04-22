.. _installing:

############
 Installing
############

The package has a split dependency structure to support different use cases:

***************************
 Basic Installation
***************************

For workflow creation without local execution:

.. code:: bash

   pip install earthkit-workflows-anemoi

This installs only the core dependencies needed to define and serialize workflows. This is useful when workflows will be executed remotely or when you only need to construct workflow definitions.

***************************
 Runtime Installation
***************************

For both workflow creation and local execution:

.. code:: bash

   pip install 'earthkit-workflows-anemoi[runtime]'

This includes the optional runtime dependencies:

- ``anemoi-inference>=0.7`` - Required for running inference tasks
- ``anemoi-datasets>=0.5.21`` - Required for dataset creation tasks

***************************
 Development Installation
***************************

For contributing to the project:

.. code:: bash

   git clone https://github.com/ecmwf/earthkit-workflows-anemoi.git
   cd earthkit-workflows-anemoi
   pip install -e '.[dev]'

This installs the package in editable mode with all development dependencies including tests and documentation tools.

.. note::
   The split dependency structure allows you to create workflows in environments where the full runtime dependencies (which can be large) are not needed, while still being able to serialize and submit workflows for remote execution.
