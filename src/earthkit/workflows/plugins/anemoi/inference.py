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
from typing import TYPE_CHECKING

import earthkit.data as ekd
from anemoi.inference.types import State
from earthkit.data.utils.dates import to_datetime

from earthkit.workflows import mark

from .runner import CascadeRunner
from .types import ENSEMBLE_DIMENSION_NAME

if TYPE_CHECKING:
    from .types import DATE, LEAD_TIME

LOG = logging.getLogger(__name__)


def _get_initial_conditions(config: dict, date: DATE, number: int | None = None) -> dict[str, State]:
    """Get initial conditions for the model"""
    runner = CascadeRunner(**config)

    states = {}
    from anemoi.inference.inputs.mars import MarsInput

    # TODO: Replace with a prefetch of all data in the case of dynamics and model uses GribInput during run
    # Use pipes to read and write
    def _mars_kwargs(input_obj):
        if isinstance(input_obj, MarsInput) and number is not None:
            return {"number": number}
        return {}

    for key in runner.dataset_names:
        dt = to_datetime(date)
        states[key] = runner._combine_states(
            runner.prognostics_inputs[key].create_input_state(date=dt, **_mars_kwargs(runner.prognostics_inputs[key])),
            runner.constant_forcings_inputs[key].create_input_state(
                date=dt, **_mars_kwargs(runner.constant_forcings_inputs[key])
            ),
            runner.dynamic_forcings_inputs[key].create_input_state(
                date=dt, **_mars_kwargs(runner.dynamic_forcings_inputs[key])
            ),
        )
        if number is not None:
            states[key][ENSEMBLE_DIMENSION_NAME] = number
        states[key].pop("_grib_templates_for_output", None)
    return states


@mark.needs_gpu
def run_as_earthkit(
    input_states: dict,
    config: dict,
    lead_time: LEAD_TIME,
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

    Returns
    -------
    Generator[dict[str, ekd.SimpleFieldList], None, None]
        State of the model at each time step
    """
    runner = CascadeRunner(**config)

    yield from runner.run(input_states=input_states, lead_time=lead_time)
    del runner.model


@mark.needs_gpu
def collect_as_earthkit(input_state: dict, config: dict, lead_time: LEAD_TIME) -> dict[str, ekd.SimpleFieldList]:
    fields: dict[str, list] = {}
    for state in run_as_earthkit(input_state, config, lead_time):
        for dataset, fieldlist in state.items():
            fields.setdefault(dataset, []).extend(fieldlist.fields)

    return {dataset: ekd.SimpleFieldList(flist) for dataset, flist in fields.items()}
