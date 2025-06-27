.. _earthkit-workflows-anemoi:

.. _index-page:

#######################################################
 Welcome to `earthkit-workflows-anemoi` documentation!
#######################################################

.. warning::

   This documentation is work in progress.

The `earthkit-workflows-anemoi` package provides a framework for anemoi
workflows utilising the `earthkit-workflows` package.

****************
 Quick overview
****************

Earthkit-Workflows-Anemoi is a Python library for connecting
`anemoi-inference <https://github.com/ecmwf/anemoi-inference>`_ to
`earthkit-workflows <https://github.com/ecmwf/earthkit-workflows>`_.
This allows inference tasks to be run as part of a larger DAG. It
provides an API to directly create a graph consisting of initial
condition retrieval and model execution, or to run inference off other
source nodes which themselves are the initial conditions.

************
 Installing
************

To install the package, you can use the following command:

.. code:: bash

   pip install earthkit-workflows-anemoi

Get more information in the :ref:`installing <installing>` section.

**************
 Contributing
**************

.. code:: bash

   git clone ...
   cd earthkit-workflows-anemoi
   pip install .[dev]

*********
 License
*********

*earthkit-workflows-anemoi* is available under the open source `Apache
License`__.

.. __: http://www.apache.org/licenses/LICENSE-2.0.html

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Introduction

   overview
   installing

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Usage

   usage/inference

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API

   api/fluent
   api/inference
