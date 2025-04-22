# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import functools
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any
from typing import Generator

from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.grib import shortname_to_paramid
from earthkit.data import ArrayField
from earthkit.data import FieldList
from earthkit.data import SimpleFieldList

from anemoi.cascade.runner import CascadeRunner

if TYPE_CHECKING:
    from anemoi.inference.config import Configuration
    from anemoi.transform.variables import Variable


def paramId_to_units(paramId: int) -> str:
    from eccodes import codes_get
    from eccodes import codes_grib_new_from_samples
    from eccodes import codes_release
    from eccodes import codes_set

    gid = codes_grib_new_from_samples("GRIB2")

    codes_set(gid, "paramId", paramId)
    units = codes_get(gid, "units")
    codes_release(gid)
    return units


def run(input_state: dict, runner: CascadeRunner, lead_time: int) -> Generator[Any, None, None]:
    """
    Run the model

    Parameters
    ----------
    input_state : dict
        Initial conditions for the model
    runner : CascadeRunner
        CascadeRunner object
    lead_time : int
        Lead time for the model

    Returns
    -------
    Generator[Any, None, None]
        State of the model at each time step
    """
    yield from runner.run(input_state=input_state, lead_time=lead_time)


def run_as_earthkit(
    input_state: dict, runner: CascadeRunner, lead_time: Any, extra_metadata: dict[str, Any] = None
) -> Generator[SimpleFieldList, None, None]:
    """
    Run the model and yield the results as earthkit FieldList

    Parameters
    ----------
    input_state : dict
        Initial Conditions for the model
    runner : CascadeRunner
        CascadeRunner Object
    lead_time : Any
        Lead time for the model
    extra_metadata: dict[str, Any], optional
        Extra metadata to add to the fields, by default None

    Returns
    -------
    Generator[SimpleFieldList, None, None]
        State of the model at each time step
    """
    initial_date: datetime = input_state["date"]
    ensemble_member = input_state.get("ensemble_member")
    extra_metadata = extra_metadata or {}

    variables: dict[str, Variable] = runner.checkpoint.typed_variables

    for state in run(input_state, runner, lead_time):
        fields = []
        step = frequency_to_seconds(state["date"] - initial_date) // 3600

        for field in state["fields"]:
            array = state["fields"][field]
            if "_grib_templates_for_output" in state and field in state["_grib_templates_for_output"]:
                metadata = state["_grib_templates_for_output"][field].metadata()
                metadata = metadata.override(
                    {"step": step, "ensemble_member": ensemble_member, **extra_metadata}, headers_only_clone=False
                )  # 'date': time_to_grib(initial_date), 'time': time_to_grib(initial_date)

            else:
                var = variables[field]

                metadata = {}
                paramId = shortname_to_paramid(var.grib_keys["param"])

                metadata.update(
                    {
                        "step": step,
                        "base_datetime": initial_date,
                        "valid_datetime": state["date"],
                        "shortName": var.name,
                        "short_name": var.name,
                        "paramId": paramId,
                        "levtype": var.grib_keys["levtype"],
                        "latitudes": runner.checkpoint.latitudes,
                        "longitudes": runner.checkpoint.longitudes,
                        "member": ensemble_member,
                        "units": paramId_to_units(paramId),
                        "values": array,  # TODO Remove
                        **extra_metadata,
                    }
                )

            fields.append(ArrayField(array, metadata))

        yield FieldList.from_fields(fields)
    del runner.model


@functools.wraps(run_as_earthkit)
def run_as_earthkit_from_config(
    input_state: dict, config: Configuration, lead_time: Any, extra_metadata: dict[str, Any] = None
) -> Generator[SimpleFieldList, None, None]:
    """Run from config"""
    runner = CascadeRunner(config)
    yield from run_as_earthkit(input_state, runner, lead_time, extra_metadata)


def collect_as_earthkit(
    input_state: dict, runner: CascadeRunner, lead_time: Any, extra_metadata: dict[str, Any] = None
) -> SimpleFieldList:
    """
    Collect the results of the model run as earthkit FieldList

    Parameters
    ----------
    input_state : dict
        Initial conditions for the model
    runner : CascadeRunner
        CascadeRunner object
    lead_time : Any
        Lead time for the model
    extra_metadata: dict[str, Any], optional
        Extra metadata to add to the fields, by default None

    Returns
    -------
    SimpleFieldList
        Combined FieldList of the model run
    """
    fields = []
    for state in run_as_earthkit(input_state, runner, lead_time, extra_metadata):
        fields.extend(state.fields)

    return SimpleFieldList(fields)


@functools.wraps(collect_as_earthkit)
def collect_as_earthkit_from_config(
    input_state: dict, config: Configuration, lead_time: Any, extra_metadata: dict[str, Any] = None
) -> SimpleFieldList:
    """Run from config"""
    runner = CascadeRunner(config)
    return collect_as_earthkit(input_state, runner, lead_time, extra_metadata)
