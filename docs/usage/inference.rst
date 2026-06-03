###########
 Inference
###########

Running inference with ``workflows-anemoi`` is straightforward and can
be done in two main ways: from an input source or from an existing
source node. Below are examples of both methods.

************
 From Input
************

To create a workflow representation of an ``anemoi-inference`` task, use
the following code:

.. code:: python

   from earthkit.workflows.plugins import anemoi as anemoi_workflows

   CKPT = {"huggingface": "ecmwf/aifs-single-1.0"}

   inference = anemoi_workflows.fluent.Inference(CKPT, lead_time="7D")
   model_action = inference.from_input("mars", "2022-01-01T00:00")
   model_action

This will load the checkpoint and use the ``mars`` input source, with a
lead time of 7 days. It is possible to configure the input source just
like you would do with the ``anemoi-inference`` interfaces.

.. note::

   The resulting action includes a ``dataset`` dimension for models with
   multiple datasets (supported since ``anemoi-inference`` 0.11.0). Use
   ``.select({"dataset": "era5"})`` to select a specific dataset if
   needed. See :doc:`/usage/expansion` for more details.

*************
 From Source
*************

If more complex initial conditions are required, you can use the
``from_initial_conditions`` method on an ``Inference`` instance.

.. code:: python

   from earthkit.workflows.plugins import anemoi as anemoi_workflows
   from earthkit.workflows import fluent

   SOURCE_NODES: fluent.Action
   CKPT = {"huggingface": "ecmwf/aifs-single-1.0"}

   inference = anemoi_workflows.fluent.Inference(CKPT, lead_time="7D")
   inference.from_initial_conditions(SOURCE_NODES)

This will use the existing source nodes as the initial conditions, and
run the inference task with the specified checkpoint, lead time. If the
source nodes have an ensemble dimension, it will also run the inference
task for each ensemble member.

****************************
 Getting Initial Conditions
****************************

If you need to retrieve the initial conditions separately — for example,
to inspect them, share them across multiple models, or compose a custom
workflow — use the
:func:`~earthkit.workflows.plugins.anemoi.fluent.get_initial_conditions`
function:

.. code:: python

   from earthkit.workflows.plugins.anemoi import fluent as anemoi_fluent

   CKPT = {"huggingface": "ecmwf/aifs-single-1.0"}

   ic = anemoi_fluent.get_initial_conditions(
       CKPT,
       "mars",
       date="2022-01-01T00:00",
   )

This returns a :class:`~earthkit.workflows.fluent.Action` representing
the initial state. You can then feed it into an inference run:

.. code:: python

   inference = anemoi_fluent.Inference(CKPT, lead_time="7D")
   forecast = inference.from_initial_conditions(ic)

This two-step approach is equivalent to using
:func:`~earthkit.workflows.plugins.anemoi.fluent.from_input`, but gives
you explicit access to the initial conditions action before inference
begins.
