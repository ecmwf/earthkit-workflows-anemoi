# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import datetime
import functools
import logging
from collections.abc import Generator
from io import BytesIO
from typing import TYPE_CHECKING, Any

import earthkit.data as ekd
from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.types import State
from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.grib import shortname_to_paramid
from earthkit.data.utils.dates import to_datetime

from earthkit.workflows import mark

from .runner import CascadeRunner

if TYPE_CHECKING:
    from anemoi.inference.input import Input
    from anemoi.transform.variables import Variable

    from .types import DATE, LEAD_TIME

LOG = logging.getLogger(__name__)


def _get_initial_conditions(input: Input, date: DATE) -> State:
    """Get initial conditions for the model"""
    input_state = input.create_input_state(date=to_datetime(date))
    assert isinstance(input_state, dict), "Input state must be a dictionary"
    return input_state


def _get_initial_conditions_ens(input: Input, ens_mem: int, date: DATE) -> State:
    """Get initial conditions for the model"""
    from anemoi.inference.inputs.mars import MarsInput

    if isinstance(input, MarsInput):  # type: ignore
        input.kwargs["number"] = ens_mem  # type: ignore

    input_state = input.create_input_state(date=to_datetime(date))
    assert isinstance(input_state, dict), "Input state must be a dictionary"
    input_state["ensemble_member"] = ens_mem

    return input_state


def _get_initial_conditions_from_config(config: RunConfiguration, date: DATE, ens_mem: int | None = None) -> State:
    """Get initial conditions for the model"""
    # TODO: Instantiate the input directly
    runner = CascadeRunner(config)
    input = runner.create_input()

    if ens_mem is not None:
        state = _get_initial_conditions_ens(input, ens_mem, date)

    state = _get_initial_conditions(input, date)
    state.pop("_grib_templates_for_output", None)
    return state


def run(input_state: dict, runner: CascadeRunner, lead_time: LEAD_TIME) -> Generator[Any]:
    """
    Run the model.

    Parameters
    ----------
    input_state : dict
        Initial conditions for the model
    runner : CascadeRunner
        CascadeRunner object
    lead_time : LEAD_TIME
        Lead time for the model

    Returns
    -------
    Generator[Any, None, None]
        State of the model at each time step
    """
    yield from runner.run(input_state=input_state, lead_time=lead_time)


def convert_to_fieldlist(
    state: dict,
    initial_date: datetime.datetime,
    runner: CascadeRunner,
    ensemble_member: int | None,
    **kwargs,
) -> ekd.SimpleFieldList:
    """
    Convert the state to an earthkit FieldList.

    Parameters
    ----------
    state :
        State of the model at a given time step
    initial_date : datetime.datetime
        Initial date of the model run
    runner : CascadeRunner
        Runner object
    ensemble_member : int | None
        Ensemble member number
    kwargs : dict
        Additional metadata to add to the fields

    Returns
    -------
    ekd.FieldList
        Earthkit FieldList with the model results
    """

    metadata = {}

    metadata.update(
        {
            "edition": 2,
            "type": "fc",
            "class": "ai",
        }
    )
    if ensemble_member is not None:
        metadata.update(
            {
                "productDefinitionTemplateNumber": 1,
                "type": "pf",
                "stream": "enfo",
                "number": ensemble_member,
                # "model" : runner.config.description or f"ai-{str(runner.config.checkpoint)}",
            }
        )
    metadata.update(kwargs)

    try:
        from anemoi.inference.outputs.gribmemory import GribMemoryOutput

        output_kwargs = runner.config.output
        if isinstance(output_kwargs, str):
            output_kwargs = {}
        if isinstance(output_kwargs, dict):
            output_kwargs = output_kwargs.copy().get("out", {})

        target = BytesIO()
        output = GribMemoryOutput(runner, out=target, encoding=metadata, **output_kwargs)
        output.write_state(state)

        target.seek(0, 0)
        fieldlist: ekd.SimpleFieldList = ekd.from_source("stream", target, read_all=True)  # type: ignore
        return fieldlist

    except Exception:
        LOG.error("Error converting state to grib, will convert to ArrayField.", exc_info=True)

    import numpy as np

    fields = []

    step = frequency_to_seconds(state["date"] - initial_date) // 3600
    variables: dict[str, Variable] = runner.checkpoint.typed_variables

    for var, array in state["fields"].items():
        variable = variables[var]
        paramId = shortname_to_paramid(variable.param)

        metadata.update(
            {
                "step": step,
                "base_datetime": initial_date,
                "valid_datetime": state["date"],
                "paramId": paramId,
                "shortName": variable.param,
                "param": variable.param,
                "latitudes": state["latitudes"],
                "longitudes": np.where(state["longitudes"] > 180, state["longitudes"] - 360, state["longitudes"]),
            }
        )
        if "levtype" in variable.grib_keys:
            metadata["levtype"] = variable.grib_keys["levtype"]
        if variable.level is not None:
            metadata["level"] = variable.level

        fields.append(ekd.ArrayField(array, metadata.copy()))

    return ekd.SimpleFieldList.from_fields(fields)


@mark.needs_gpu
def run_as_earthkit(
    input_state: dict, runner: CascadeRunner, lead_time: LEAD_TIME, extra_metadata: dict[str, Any] | None = None
) -> Generator[ekd.SimpleFieldList]:
    """
    Run the model and yield the results as earthkit FieldList

    Parameters
    ----------
    input_state : dict
        Initial Conditions for the model
    runner : CascadeRunner
        CascadeRunner Object
    lead_time : LEAD_TIME
        Lead time for the model
    extra_metadata: dict[str, Any], optional
        Extra metadata to add to the fields, by default None

    Returns
    -------
    Generator[SimpleFieldList, None, None]
        State of the model at each time step
    """

    initial_date: datetime.datetime = input_state["date"]
    ensemble_member = input_state.get("ensemble_member", None)
    extra_metadata = extra_metadata or {}

    post_processors = runner.create_post_processors()

    for state in run(input_state, runner, lead_time):
        for processor in post_processors:
            state = processor.process(state)

        yield convert_to_fieldlist(
            state,
            initial_date,
            runner,
            ensemble_member=ensemble_member,
            **extra_metadata,
        )

    del runner.model


@functools.wraps(run_as_earthkit)
@mark.needs_gpu
def run_as_earthkit_from_config(
    input_state: dict,
    config: RunConfiguration,
    **kw,
) -> Generator[ekd.SimpleFieldList]:
    runner = CascadeRunner(config)
    yield from run_as_earthkit(input_state, runner, **kw)


@mark.needs_gpu
def collect_as_earthkit(
    input_state: dict, runner: CascadeRunner, lead_time: LEAD_TIME, extra_metadata: dict[str, Any] | None = None
) -> ekd.SimpleFieldList:
    """
    Collect the results of the model run as earthkit FieldList

    Parameters
    ----------
    input_state : dict
        Initial conditions for the model
    runner : CascadeRunner
        CascadeRunner object
    lead_time : LEAD_TIME
        Lead time for the model
    extra_metadata: dict[str, Any], optional
        Extra metadata to add to the fields, by default None

    Returns
    -------
    ekd.SimpleFieldList
        Combined FieldList of the model run
    """
    fields = []
    for state in run_as_earthkit(input_state, runner, lead_time, extra_metadata):
        fields.extend(state.fields)

    return ekd.SimpleFieldList(fields)


@functools.wraps(collect_as_earthkit)
@mark.needs_gpu
def collect_as_earthkit_from_config(input_state: dict, config: RunConfiguration, **kw) -> ekd.SimpleFieldList:
    runner = CascadeRunner(config)
    return collect_as_earthkit(input_state, runner, **kw)
