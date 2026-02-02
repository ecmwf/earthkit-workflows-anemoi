#####################
 Expansion Utilities
#####################

The expansion utilities in ``earthkit-workflows-anemoi`` provide useful
tools for creating and manipulating Expansion objects based on model
metadata.

****************
 What is Qubed?
****************

The expansion system is built on `Qubed
<https://github.com/ecmwf/qubed>`_, a library for representing
multi-dimensional data structures. A Qube defines the axes (dimensions)
and their values, along with optional hierarchical relationships between
different sets of dimensions.

Think of a Qube as a blueprint that describes how your actions should be
distributed across different coordinates:

-  **Simple Qube**: Single set of dimensions (e.g., time steps)
-  **Hierarchical Qube**: Multiple branches with different dimension
   sets (e.g., surface variables vs. pressure level variables)

**************************
 Basic Expansion Workflow
**************************

Using Expansion with Model Metadata
===================================

The most common use case is to create an expansion directly from model
metadata using the ``expansion_coordinates`` function:

.. code:: python

   from earthkit.workflows.plugins.anemoi.utils import expansion_coordinates
   from anemoi.inference.checkpoint import Checkpoint

   # Load model checkpoint
   ckpt = Checkpoint("path/to/checkpoint.ckpt")
   metadata = ckpt.metadata

   # Create expansion for a 5-day forecast
   expansion = expansion_coordinates(metadata, lead_time="5D")

   # View the dimensions
   print(expansion.axes())
   # Output: {'step': {6, 12, 18, ..., 120}, 'param': {...}, 'level': {...}}

   # Expand an action across all dimensions
   expanded_action = expansion.expand(action)

This automatically organises variables into surface, pressure, and model
level groups, each expanded across the appropriate dimensions.

***************************
 Manual Expansion Creation
***************************

For more control, you can manually construct Expansion objects with
Qubed:

Simple Single-Dimension Expansion
=================================

.. code:: python

   from qubed import Qube
   from earthkit.workflows.plugins.anemoi.utils import Expansion

   # Create a simple qube with time steps
   qube = Qube.from_datacube({"step": [6, 12, 18, 24]})
   expansion = Expansion(qube)

   # Expand an action
   expanded_action = expansion.expand(action)

Multi-Dimensional Expansion
===========================

.. code:: python

   # Create a qube with multiple dimensions
   qube = Qube.from_datacube({
       "step": [6, 12, 18],
       "param": ["t", "q", "u", "v"],
       "level": [500, 850, 1000]
   })
   expansion = Expansion(qube)
   expanded_action = expansion.expand(action)

Hierarchical Expansion
======================

Create separate expansions for different variable types:

.. code:: python

   from qubed import Qube
   from earthkit.workflows.plugins.anemoi.utils import Expansion

   # Surface variables (2D fields)
   surface = Qube.from_datacube({
       "param": ["2t", "2d", "10u", "10v", "msl"]
   })
   surface.add_metadata({"name": "surface"})

   # Pressure level variables (3D fields)
   pressure = Qube.from_datacube({
       "param": ["t", "q", "u", "v"],
       "level": [500, 700, 850, 925, 1000]
   })
   pressure.add_metadata({"name": "pressure"})

   # Combine with time steps
   steps = Qube.from_datacube({"step": [6, 12, 18, 24]})
   combined = steps | (surface | pressure)

   # Create expansion
   expansion = Expansion(combined)
   expanded_action = expansion.expand(action)

The expanded action will have separate branches for ``/surface`` and
``/pressure``, each containing the appropriate parameters and levels.

**********************
 Modifying Expansions
**********************

Dropping Axes
=============

You can remove dimensions from an expansion before applying it:

.. code:: python

   # Create expansion with multiple dimensions
   qube = Qube.from_datacube({
       "step": [6, 12, 18],
       "param": ["t", "q"],
       "level": [500, 850, 1000]
   })
   expansion = Expansion(qube)

   # Drop the time step dimension
   no_steps = expansion.drop_axis("step")
   expanded = no_steps.expand(action)

   # Drop multiple dimensions
   params_only = expansion.drop_axis(["step", "level"])
   expanded = params_only.expand(action)

Inspecting Axes
===============

View available dimensions before expansion:

.. code:: python

   expansion = Expansion(qube)
   axes = expansion.axes()

   for axis_name, values in axes.items():
       print(f"{axis_name}: {sorted(values)}")

   # Check if an axis exists
   if "level" in axes:
       print(f"Pressure levels: {sorted(axes['level'])}")

*************
 API Summary
*************

**Expansion Class**

-  ``Expansion(qube)``: Create an expansion from a Qube
-  ``expansion.axes()``: View available dimensions
-  ``expansion.drop_axis(axis)``: Remove dimension(s)
-  ``expansion.expand(action)``: Apply expansion to action

**Helper Functions**

-  ``expansion_coordinates(metadata, lead_time)``: Create expansion from
   model metadata

**See Also**

-  :doc:`/usage/inference` - Using expansions with inference workflows
-  :doc:`/api/fluent` - Fluent API documentation
-  `Qubed Documentation <https://qubed.readthedocs.io/>`_ - Underlying
   data structure library
