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

# from earthkit.data.utils.dates import time_to_grib, date_to_grib


if TYPE_CHECKING:
    from anemoi.inference.config import Configuration
    from anemoi.transform.variables import Variable


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


def run_as_earthkit(input_state: dict, runner: CascadeRunner, lead_time: Any) -> Generator[SimpleFieldList, None, None]:
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

    Returns
    -------
    Generator[SimpleFieldList, None, None]
        State of the model at each time step
    """
    initial_date: datetime = input_state["date"]

    variables: dict[str, Variable] = runner.checkpoint.typed_variables

    for state in run(input_state, runner, lead_time):
        fields = []
        step = frequency_to_seconds(state["date"] - initial_date) // 3600

        for field in state["fields"]:
            array = state["fields"][field]
            if "_grib_templates_for_output" in state and field in state["_grib_templates_for_output"]:
                metadata = state["_grib_templates_for_output"][field].metadata()
                metadata = metadata.override(
                    {"step": step}, headers_only_clone=False
                )  # 'date': time_to_grib(initial_date), 'time': time_to_grib(initial_date)

            else:
                var = variables[field]
                metadata = var.grib_keys
                metadata.update(
                    {
                        "step": step,
                        "shortName": var.name,
                        "paramId": shortname_to_paramid(metadata["param"]),
                        "base_datetime": initial_date,
                        "latitudes": runner.checkpoint.latitudes,
                        "longitudes": runner.checkpoint.longitudes,
                        "values": array,  # TODO Remove
                    }
                )

            fields.append(ArrayField(array, metadata))
            fields[-1].metadata().geography

        yield FieldList.from_fields(fields)


@functools.wraps(run_as_earthkit)
def run_as_earthkit_from_config(
    input_state: dict, config: Configuration, lead_time: Any
) -> Generator[SimpleFieldList, None, None]:
    """Run from config"""
    runner = CascadeRunner(config)
    yield from run_as_earthkit(input_state, runner, lead_time)


def collect_as_earthkit(input_state: dict, runner: CascadeRunner, lead_time: Any) -> SimpleFieldList:
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

    Returns
    -------
    SimpleFieldList
        Combined FieldList of the model run
    """
    fields = []
    for state in run_as_earthkit(input_state, runner, lead_time):
        fields.extend(state.fields)

    return SimpleFieldList(fields)
