.. _overview:

##########
 Overview
##########

`anemoi-inference <https://github.com/ecmwf/anemoi-inference>`_ to
`earthkit-workflows <https://github.com/ecmwf/earthkit-workflows>`_.
This allows inference tasks to be run as part of a larger DAG. It
provides an API to directly create a graph consisting of initial
condition retrieval and model execution, or to run inference off other
source nodes which themselves are the initial conditions.

***********************
 Architecture
***********************

The package is designed with a split dependency structure:

**Core Dependencies**

The base installation includes only:

- ``anemoi-utils`` - Utility functions
- ``earthkit-workflows>=0.7`` - Workflow framework
- ``qubed>=0.3`` - Multi-dimensional data structure support

**Runtime Dependencies (Optional)**

For local workflow execution, install with ``[runtime]`` extra:

- ``anemoi-inference>=0.7`` - AI weather model inference
- ``anemoi-datasets>=0.5.21`` - Dataset creation and management

This separation allows:

1. **Lightweight workflow creation**: Create and serialize workflows without heavy ML dependencies
2. **Remote execution**: Define workflows on lightweight machines, execute on compute infrastructure
3. **Flexible deployment**: Install only what you need for your use case
