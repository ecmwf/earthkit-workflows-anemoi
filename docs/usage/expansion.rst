###############################
 Understanding Expansion Qubes
###############################

The ``expansion_qube`` utility in ``earthkit-workflows-anemoi`` creates
hierarchical Qube structures from model metadata. These qubes help you
understand how your forecast data is organised across dimensions like
time steps, parameters, and vertical levels.

****************
 What is Qubed?
****************

`Qubed <https://github.com/ecmwf/qubed>`_ is a library for representing
multi-dimensional data structures. A Qube defines axes (dimensions) and
their values, along with hierarchical relationships between different
sets of dimensions.

Think of a Qube as a blueprint showing how data is organised:

-  **Simple Qube**: Single set of dimensions (e.g., time steps)
-  **Hierarchical Qube**: Multiple branches with different dimension
   sets (e.g., surface variables separate from pressure level variables)

*****************************
 The expansion_qube Function
*****************************

The ``expansion_qube`` function automatically analyses model metadata
and creates a hierarchical Qube structure, organising variables by their
vertical coordinate type.

.. note::

   Since version 0.11.0 of ``anemoi-inference``, models support multiple
   datasets. The ``expansion_qube_from_metadata`` function now returns a
   **dictionary of Qubes** (``dict[str, Qube]``), where each key
   represents a dataset name (e.g., ``"era5"``, ``"cerra"``,
   ``"data"``).

Basic Usage
===========

.. code:: python

   from earthkit.workflows.plugins.anemoi.utils import expansion_qube_from_metadata
   from anemoi.inference.checkpoint import Checkpoint

   # Load model checkpoint
   ckpt = Checkpoint("path/to/checkpoint.ckpt")

   # Create qube dictionary for a 5-day forecast
   qubes = expansion_qube_from_metadata(ckpt.multi_dataset_metadata, lead_time="5D")

   # qubes is now a dict[str, Qube]
   print(qubes.keys())  # e.g., dict_keys(['data'])

   # Inspect a specific dataset's qube structure
   qube = qubes["data"]
   print(qube.axes())
   # Output: {'step': [6, 12, 18, ..., 120],
   #          'param': ['2t', '10u', ...],
   #          'level': [500, 850, 1000, ...],
   #          'levtype': ['sfc', 'pl', 'ml']}

What the Qube Represents
========================

The returned dictionary of qubes describes how forecast data will be
organised when you use the anemoi fluent API. Each dataset in the
dictionary contains up to three named branches:

#. **surface**: Surface-level 2D fields

   -  Dimensions: ``step``, ``param``, ``levtype`` (='sfc')
   -  Examples: 2-metre temperature (``2t``), 10-metre winds (``10u``,
      ``10v``)

#. **pressure**: Pressure-level 3D fields

   -  Dimensions: ``step``, ``param``, ``level``, ``levtype`` (='pl')
   -  Examples: Temperature (``t``), specific humidity (``q``) at
      various pressure levels

#. **model**: Model-level 3D fields

   -  Dimensions: ``step``, ``param``, ``level``, ``levtype`` (='ml')
   -  Examples: Variables on native model vertical coordinates

The time steps are calculated automatically from the model's native time
step up to the specified lead time.

**Note**: When you use functions like ``from_input`` or
``from_initial_conditions``, the expansion is applied automatically
using this structure, and a ``dataset`` dimension is added to the
resulting workflow action.

**************************
 Multi-Dataset Dimensions
**************************

When working with models that support multiple datasets (introduced in
``anemoi-inference`` 0.11.0), the resulting workflow actions include a
``dataset`` dimension. This dimension allows you to distinguish between
different datasets in the model output.

Understanding the Dataset Dimension
===================================

When you create an inference action, the ``dataset`` dimension is
automatically added based on the keys in the expansion qube dictionary:

.. code:: python

   from earthkit.workflows.plugins.anemoi.fluent import from_input

   # Create an inference action
   action = from_input(
       ckpt="path/to/checkpoint.ckpt",
       input="mars",
       date="2022-01-01T00:00",
       lead_time="7D"
   )

   # The action now has a dataset dimension
   print(action.nodes.dims)
   # Output: ('date', 'number', 'step', 'dataset', ...)
   # Check dataset values
   print(action.nodes.coords["dataset"].values)
   # Output: ['era5'] (or multiple dataset names if the model has them)

Selecting a Specific Dataset
============================

To work with a specific dataset, use the ``.select()`` method:

.. code:: python

   # Select only the 'era5' dataset
   era5_only = action.select({"dataset": "era5"})

   # Now the dataset dimension is removed
   print(era5_only.nodes.dims)
   # Output: ('date', 'number', 'step', ...)
For models with multiple datasets (e.g., ERA5 and CERRA datasets):

.. code:: python

   # If the model has multiple datasets
   print(action.nodes.coords["dataset"].values)
   # Output: ['era5', 'cerra']

   # Select only ERA5 data
   era5_data = action.select({"dataset": "era5"})

   # Select only CERRA data
   cerra_data = action.select({"dataset": "cerra"})

Single Qube Behaviour
=====================

When you explicitly provide a **single Qube** (not a dictionary) to the
``expansion_qube`` parameter, the dataset dimension is automatically
selected away:

.. code:: python

   from qubed import Qube
   from earthkit.workflows.plugins.anemoi.fluent import Inference

   # Create a custom single qube
   qube = Qube.from_datacube({
       "step": [6, 12, 18, 24],
       "param": ["2t", "msl"]
   })

   # Use it in inference
   inference = Inference(ckpt, lead_time="1D", expansion_qube=qube)
   action = inference.from_input("mars", date="2022-01-01")

   # The dataset dimension is NOT present
   print("dataset" in action.nodes.dims)
   # Output: False

This is for backwards compatibility and convenience when working with
simple single-dataset models.

Multiple Dataset Example
========================

For advanced use cases with multiple datasets:

.. code:: python

   from qubed import Qube
   from earthkit.workflows.plugins.anemoi.fluent import from_initial_conditions

   # Define separate qubes for different datasets
   era5_qube = Qube.from_datacube({
       "step": [6, 12, 18, 24],
       "param": ["2t", "10u", "10v"],
       "levtype": ["sfc"]
   })

   cerra_qube = Qube.from_datacube({
       "step": [6, 12, 18, 24],
       "param": ["t", "q", "u", "v"],
       "level": [500, 850, 1000],
       "levtype": ["pl"]
   })

   # Combine into multi-dataset dictionary
   multi_qube = {
       "era5": era5_qube,
       "cerra": cerra_qube
   }

   # Create inference action
   action = from_initial_conditions(
       ckpt="path/to/checkpoint.ckpt",
       initial_conditions=None,
       lead_time="1D",
       expansion_qube=multi_qube
   )

   # Dataset dimension is present with both keys
   print(action.nodes.coords["dataset"].values)
   # Output: ['era5', 'cerra']

   # Select each dataset separately
   era5 = action.select({"dataset": "era5"})
   cerra = action.select({"dataset": "cerra"})

******************
 Inspecting Qubes
******************

Understanding what's in a qube helps you know what data to expect in
your forecast actions.

View All Dimensions
===================

.. code:: python

   from earthkit.workflows.plugins.anemoi.utils import expansion_qube
   from anemoi.inference.checkpoint import Checkpoint

   ckpt = Checkpoint("path/to/checkpoint.ckpt")
   qube = expansion_qube(ckpt.metadata, lead_time="5D")

   # View all axes and their values
   axes = qube.axes()
   for axis_name, values in axes.items():
       n_values = len(values)
       sample = sorted(values)[:5]
       print(f"{axis_name}: {n_values} values, e.g., {sample}...")

Example output:

.. code:: text

   step: 20 values, e.g., [6, 12, 18, 24, 30]...
   param: 48 values, e.g., ['2d', '2t', '10u', '10v', '100u']...
   level: 13 values, e.g., [50, 100, 150, 200, 250]...
   levtype: 3 values, e.g., ['ml', 'pl', 'sfc']...

Check Hierarchical Structure
============================

Understand how variables are grouped:

.. code:: python

   # Check number of branches
   print(f"Number of branches: {len(qube.children)}")

   # Inspect each branch
   for child in qube.children:
       if "name" in child.metadata:
           name = child.metadata["name"]
           axes = child.axes()
           print(f"\nBranch: {name}")
           print(f"  Dimensions: {list(axes.keys())}")

           # Show parameter count
           if "param" in axes:
               print(f"  Parameters: {len(axes['param'])}")

           # Show levels if present
           if "level" in axes:
               print(f"  Levels: {sorted(axes['level'])}")

Example output:

.. code:: text

   Number of branches: 3

   Branch: surface
     Dimensions: ['step', 'param', 'levtype']
     Parameters: 24

   Branch: pressure
     Dimensions: ['step', 'param', 'level', 'levtype']
     Parameters: 6
     Levels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

   Branch: model
     Dimensions: ['step', 'param', 'level', 'levtype']
     Parameters: 18
     Levels: [1, 2, 3, ..., 137]

Find Specific Variables
=======================

Check if specific parameters are present:

.. code:: python

   axes = qube.axes()
   params = axes.get("param", set())

   # Check for specific variables
   surface_vars = ["2t", "10u", "10v", "msl"]
   for var in surface_vars:
       status = "✓" if var in params else "✗"
       print(f"{status} {var}")

   # Check pressure levels
   levels = axes.get("level", set())
   required_levels = [500, 700, 850, 1000]
   available = [lev for lev in required_levels if lev in levels]
   print(f"\nAvailable levels: {available}")

*****************
 Troubleshooting
*****************

Empty Qube
==========

If ``expansion_qube`` returns an empty qube:

.. code:: python

   qube = expansion_qube(metadata, lead_time="5D")

   if not qube.children:
       print("No variables found in metadata")

       # Check what variables are available
       vars = metadata.select_variables(
           include=["diagnostic", "prognostic"],
           has_mars_requests=False
       )
       print(f"Available variables: {len(vars)}")
       print(f"Examples: {list(vars)[:10]}")

Missing Variable Types
======================

Check variable distribution across vertical coordinates:

.. code:: python

   # Inspect metadata variables
   all_vars = metadata.typed_variables

   surface_vars = [v for v in all_vars.values() if v.is_surface_level]
   pressure_vars = [v for v in all_vars.values() if v.is_pressure_level]
   model_vars = [v for v in all_vars.values() if v.is_model_level]

   print(f"Surface variables: {len(surface_vars)}")
   print(f"  Examples: {[v.param for v in surface_vars[:5]]}")

   print(f"Pressure variables: {len(pressure_vars)}")
   print(f"  Examples: {[v.param for v in pressure_vars[:5]]}")

   print(f"Model variables: {len(model_vars)}")
   print(f"  Examples: {[v.param for v in model_vars[:5]]}")

Unexpected Time Steps
=====================

Verify the time step calculation:

.. code:: python

   from anemoi.utils.dates import frequency_to_seconds

   # Check model time step
   model_step_seconds = metadata.timestep.seconds
   model_step_hours = model_step_seconds // 3600
   print(f"Model time step: {model_step_hours} hours")

   # Calculate expected number of steps
   lead_time_seconds = frequency_to_seconds("5D")
   lead_time_hours = lead_time_seconds // 3600
   n_steps = lead_time_hours // model_step_hours
   print(f"Expected steps: {n_steps}")
   print(f"Steps: {list(range(model_step_hours, lead_time_hours + 1, model_step_hours))}")

**************************
 Manual Qube Construction
**************************

For advanced use cases, you can create custom qubes manually:

Simple Qube
===========

.. code:: python

   from qubed import Qube

   # Single dimension
   qube = Qube.from_datacube({"step": [6, 12, 18, 24]})

   # Multiple dimensions
   qube = Qube.from_datacube({
       "step": [6, 12, 18],
       "param": ["t", "q", "u", "v"],
       "level": [500, 850, 1000]
   })

Hierarchical Qube
=================

.. code:: python

   # Create separate branches
   surface = Qube.from_datacube({
       "param": ["2t", "10u", "10v"],
       "levtype": ["sfc"]
   })
   surface.add_metadata({"name": "surface"})

   pressure = Qube.from_datacube({
       "param": ["t", "q", "u", "v"],
       "level": [500, 850, 1000],
       "levtype": ["pl"]
   })
   pressure.add_metadata({"name": "pressure"})

   # Combine branches
   steps = Qube.from_datacube({"step": [6, 12, 18, 24]})
   combined = steps | (surface | pressure)

   # Inspect structure
   print(f"Children: {len(combined.children)}")
   for child in combined.children:
       print(f"  {child.metadata.get('name', 'unnamed')}: {list(child.axes().keys())}")

*************
 API Summary
*************

**Main Function**

-  ``expansion_qube(metadata, lead_time)``: Create hierarchical qube
   from model metadata

**Qube Methods**

-  ``qube.axes()``: View dimensions and their values
-  ``qube.children``: Access child qubes in hierarchy
-  ``qube.metadata``: Access qube metadata

**See Also**

-  :doc:`/usage/inference` - Using the anemoi fluent API
-  :doc:`/api/utils` - API documentation for expansion_qube
-  :doc:`/api/fluent` - Fluent API documentation
-  `Qubed Documentation <https://qubed.readthedocs.io/>`_ - Underlying
   library
