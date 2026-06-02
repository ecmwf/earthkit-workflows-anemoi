# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import earthkit.data as ekd
from anemoi.inference.types import State
from earthkit.data.utils.dates import to_datetime

from earthkit.workflows import mark

from .runner import CascadeRunner
from .types import ENSEMBLE_DIMENSION_NAME

if TYPE_CHECKING:
    from .types import DATE, LEAD_TIME

LOG = logging.getLogger(__name__)


def _get_initial_conditions(config: dict, date: DATE, ens_mem: int | None = None) -> dict[str, State]:
    """Get initial conditions for the model"""
    runner = CascadeRunner(**config)

    states = {}
    from anemoi.inference.inputs.mars import MarsInput

    for key, input in runner.prognostics_inputs.items():
        if isinstance(input, MarsInput):
            input.kwargs["number"] = ens_mem

        states[key] = input.create_input_state(date=to_datetime(date))
        if ens_mem is not None:
            states[key][ENSEMBLE_DIMENSION_NAME] = ens_mem

        states[key].pop("_grib_templates_for_output", None)

    return states


@mark.needs_gpu
def run_as_earthkit(
    input_states: dict,
    config: dict,
    lead_time: LEAD_TIME,
    extra_metadata: dict[str, Any] | None = None,
) -> Generator[dict[str, ekd.SimpleFieldList]]:
    """
    Run the model and yield the results as earthkit FieldList

    Parameters
    ----------
    input_states : dict
        Initial Conditions for the model
    config : dict
        Configuration for the model run
    lead_time : LEAD_TIME
        Lead time for the model
    extra_metadata: dict[str, Any], optional
        Extra metadata to add to the fields, by default None

    Returns
    -------
    Generator[dict[str, ekd.SimpleFieldList], None, None]
        State of the model at each time step
    """
    runner = CascadeRunner(**config)

    extra_metadata = extra_metadata or {}

    for states in runner.run(input_states=input_states, lead_time=lead_time):
        # Run post-processors for each dataset
        output_states = {}
        for dataset, state in states.items():
            for processor in runner.post_processors[dataset]:
                state = processor.process(state)
            output_states[dataset] = runner.outputs[dataset].write_step(state)
        yield output_states
    del runner.model


@mark.needs_gpu
def collect_as_earthkit(
    input_state: dict, config: dict, lead_time: LEAD_TIME, extra_metadata: dict[str, Any] | None = None
) -> dict[str, ekd.SimpleFieldList]:
    fields: dict[str, ekd.SimpleFieldList] = {}
    for state in run_as_earthkit(input_state, config, lead_time, extra_metadata):
        for dataset, fieldlist in state.items():
            if dataset not in fields:
                fields[dataset] = ekd.SimpleFieldList([])
            fields[dataset].append(fieldlist)

    return fields
