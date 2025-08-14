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
import os
from io import BytesIO
from typing import TYPE_CHECKING
from typing import Any
from typing import Generator
from typing import Optional

import earthkit.data as ekd
from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.types import State
from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.grib import shortname_to_paramid
from earthkit.data.utils.dates import to_datetime

from earthkit.workflows import fluent
from earthkit.workflows import mark
from earthkit.workflows.plugins.anemoi.runner import CascadeRunner
from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

if TYPE_CHECKING:
    from anemoi.inference.input import Input
    from anemoi.transform.variables import Variable

    from earthkit.workflows.plugins.anemoi.types import DATE
    from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_MEMBER_SPECIFICATION
    from earthkit.workflows.plugins.anemoi.types import LEAD_TIME

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


def _get_initial_conditions_from_config(config: RunConfiguration, date: DATE, ens_mem: Optional[int] = None) -> State:
    """Get initial conditions for the model"""
    runner = CascadeRunner(config)
    input = runner.create_input()

    if hasattr(runner.config, "env"):
        # Set environment variables found in the configuration
        for key, value in runner.config.env.items():
            os.environ[key] = str(value)

    if ens_mem is not None and ens_mem >= 1:
        state = _get_initial_conditions_ens(input, ens_mem, date)
    state = _get_initial_conditions(input, date)
    state.pop("_grib_templates_for_output", None)
    return state


def _transform_fake(act: fluent.Action, ens_num: int) -> fluent.Action:
    """Transform the action to simulate ensemble members"""

    def _empty_payload(x, ens_mem: int):
        assert isinstance(x, dict), "Input state must be a dictionary"
        x["ensemble_member"] = ens_mem
        return x

    return act.map(fluent.Payload(_empty_payload, [fluent.Node.input_name(0), ens_num]))


def _parse_ensemble_members(ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION) -> list[int]:
    """Parse ensemble members"""
    if isinstance(ensemble_members, int):
        if ensemble_members < 1:
            raise ValueError("Number of ensemble members must be greater than 0.")
        return list(range(ensemble_members))
    return list(ensemble_members)


def get_initial_conditions_source(
    config: RunConfiguration | fluent.Action,
    date: DATE,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION = 1,
    *,
    initial_condition_perturbation: bool = False,
    payload_metadata: Optional[dict[str, Any]] = None,
) -> fluent.Action:
    """
    Get the initial conditions for the model

    Parameters
    ----------
    config : RunConfiguration | fluent.Action
        Configuration object, must contain checkpoint and input.
        If is a fluent action, the action must return the RunConfiguration object.
    date : str | tuple[int, int, int]
        Date to get initial conditions for
    ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION, optional
        Number of ensemble members to get, by default 1
    initial_condition_perturbation : bool, optional
        Whether to get perturbed initial conditions, by default False
        If False, only one initial condition is returned, and
        the ensemble members are simulated by wrapping the action.
    payload_metadata : Optional[dict[str, Any]], optional
        Metadata to add to the payload, by default None

    Returns
    -------
    fluent.Action
        Fluent action of the initial conditions
    """
    ensemble_members = _parse_ensemble_members(ensemble_members)
    if initial_condition_perturbation:
        if isinstance(config, fluent.Action):
            init_conditions = config.transform(
                lambda x, *a: x.map(
                    fluent.Payload(
                        _get_initial_conditions_from_config,
                        args=(fluent.Node.input_name(0)),
                        kwargs=dict(ens_num=a[0], date=date),
                        metadata=payload_metadata,
                    )
                ),
                params=ensemble_members,
                dim=(ENSEMBLE_DIMENSION_NAME, ensemble_members),
            )
            init_conditions._add_dimension("date", [to_datetime(date)])
            return init_conditions

        return fluent.from_source(
            [
                [
                    # fluent.Payload(_get_initial_conditions_ens, kwargs=dict(input=input, date=date, ens_mem=ens_mem))
                    fluent.Payload(
                        _get_initial_conditions_from_config,
                        kwargs=dict(config=config, date=date, ens_mem=ens_mem),
                        metadata=payload_metadata,
                    )
                    for ens_mem in ensemble_members
                ],
            ],  # type: ignore
            coords={"date": [to_datetime(date)], ENSEMBLE_DIMENSION_NAME: ensemble_members},
        )

    if isinstance(config, fluent.Action):
        init_condition = fluent.Payload(
            _get_initial_conditions_from_config,
            args=(fluent.Node.input_name(0),),
            kwargs=dict(date=date),
            metadata=payload_metadata,
        )
        single_init = config.map(init_condition)
        single_init._add_dimension("date", [to_datetime(date)])
    else:
        init_condition = fluent.Payload(
            _get_initial_conditions_from_config, kwargs=dict(config=config, date=date), metadata=payload_metadata
        )
        single_init = fluent.from_source(
            [
                init_condition,
            ],  # type: ignore
            coords={"date": [to_datetime(date)]},
        )

    # Wrap with empty payload to simulate ensemble members
    expanded_init = single_init.transform(
        _transform_fake,
        list(zip(ensemble_members)),
        (ENSEMBLE_DIMENSION_NAME, ensemble_members),  # type: ignore
    )
    if ENSEMBLE_DIMENSION_NAME not in expanded_init.nodes.coords:
        expanded_init.nodes = expanded_init.nodes.expand_dims(ENSEMBLE_DIMENSION_NAME)
    return expanded_init


def _time_range(
    start: datetime.datetime, end: datetime.datetime, step: datetime.timedelta
) -> Generator[datetime.datetime, None, None]:
    """Get a range of timedeltas"""
    while start < end:
        yield start
        start += step


def _expand(runner: CascadeRunner, model_results: fluent.Action) -> fluent.Action:
    """Expand model results into the parameter dimension"""

    # Expand by variable
    variables = [*runner.checkpoint.diagnostic_variables, *runner.checkpoint.prognostic_variables]

    # Seperate surface and pressure variables
    surface_vars = [var for var in variables if "_" not in var]

    pressure_vars_complete = [var for var in variables if "_" in var]
    # pressure_vars = list(set(var.split("_")[0] for var in variables if "_" in var))
    # pressure_levels = list(set(int(var.split('_')[1]) for var in variables if "_" in var))

    surface_expansion = None
    if surface_vars:
        surface_expansion = model_results.expand(
            ("param", surface_vars), ("param", surface_vars), backend_kwargs=dict(method="sel")
        )

    pressure_expansion = None
    if pressure_vars_complete:
        pressure_expansion = model_results.expand(
            ("param", pressure_vars_complete),
            ("param_level", pressure_vars_complete),
            backend_kwargs=dict(method="sel", remapping={"param_level": "{param}_{level}"}),
        )

        # pressure_expansion = model_results.expand(
        #     ("param", pressure_vars), ("param", pressure_vars), backend_kwargs=dict(method="sel")
        # )
        # pressure_expansion = pressure_expansion.expand(('level', pressure_levels), ('level', pressure_levels), backend_kwargs=dict(method="sel"))

    if surface_expansion is not None and pressure_expansion is not None:
        model_results = surface_expansion.join(pressure_expansion, dim="param")
    elif surface_expansion is not None:
        model_results = surface_expansion
    elif pressure_expansion is not None:
        model_results = pressure_expansion
    else:
        raise ValueError("No variables to expand")

    return model_results


def run_model(
    runner: CascadeRunner,
    config: RunConfiguration,
    input_state_source: fluent.Action,
    lead_time: LEAD_TIME,
    payload_metadata: Optional[dict[str, Any]] = None,
    **kwargs,
) -> fluent.Action:
    """
    Run the model, expanding the results to the correct dimensions.

    Parameters
    ----------
    runner : Runner
        `anemoi.inference` runner
    config : RunConfiguration
        Configuration object
    input_state_source : fluent.Action
        Fluent action of initial conditions
    lead_time : LEAD_TIME
        Lead time to run out to. Can be a string,
        i.e. `1H`, `1D`, int, or a datetime.timedelta
    payload_metadata : Optional[dict[str, Any]], optional
        Metadata to add to the payload, by default None
    kwargs : dict
        Additional arguments to pass to the runner

    Returns
    -------
    fluent.Action
        Cascade action of the model results
    """
    lead_time = to_timedelta(lead_time)

    model_payload = fluent.Payload(
        run_as_earthkit_from_config,
        args=(fluent.Node.input_name(0),),
        kwargs=dict(config=config, lead_time=lead_time, **kwargs),
        metadata=payload_metadata,
    )

    model_step = runner.checkpoint.timestep
    steps = list(
        map(lambda x: frequency_to_seconds(x) // 3600, _time_range(model_step, lead_time + model_step, model_step))
    )

    model_results = input_state_source.map(model_payload, yields=("step", steps))

    return _expand(runner, model_results)


def _paramId_to_units(paramId: int) -> str:
    """Get the units for a given paramId."""
    from eccodes import codes_get
    from eccodes import codes_grib_new_from_samples
    from eccodes import codes_release
    from eccodes import codes_set

    gid = codes_grib_new_from_samples("GRIB2")

    codes_set(gid, "paramId", paramId)
    units = codes_get(gid, "units")
    codes_release(gid)
    return str(units)


def run(input_state: dict, runner: CascadeRunner, lead_time: LEAD_TIME) -> Generator[Any, None, None]:
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
    templates: list[str] | None = None,
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
    templates: list[str] | None
        Templates to use for the output, by default None
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
    if ensemble_member is not None and ensemble_member >= 1:
        metadata.update(
            {
                "type": "pf",
                "stream": "enfo",
                "number": ensemble_member,
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
        output = GribMemoryOutput(runner, out=target, grib2_keys=metadata, **output_kwargs)
        output.write_state(state)

        target.seek(0, 0)
        fieldlist: ekd.SimpleFieldList = ekd.from_source("stream", target, read_all=True)  # type: ignore
        return fieldlist

    except Exception as e:
        LOG.warning(f"Error converting state to grib, will convert to ArrayField. {e}")

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
                "units": _paramId_to_units(paramId),
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
) -> Generator[ekd.SimpleFieldList, None, None]:
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
    if hasattr(runner.config, "env"):
        # Set environment variables found in the configuration
        for key, value in runner.config.env.items():
            os.environ[key] = str(value)

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
) -> Generator[ekd.SimpleFieldList, None, None]:
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
