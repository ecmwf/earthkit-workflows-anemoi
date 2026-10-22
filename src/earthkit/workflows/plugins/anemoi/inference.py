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


def _member_request(request: dict, number: int) -> dict:
    """Turn a MARS request into the initial-condition request of ensemble member ``number``.

    Member 0 is the control forecast (``stream=enfo, type=cf``, never a ``number``), any other member is a
    perturbed forecast (``stream=enfo, type=pf, number=<member>``). Both are the analysis time, ``step=0``.
    """
    request = dict(request)
    request["stream"] = "enfo"
    request["step"] = 0
    if number == 0:
        request["type"] = "cf"
        request.pop("number", None)
    else:
        request["type"] = "pf"
        request["number"] = number
    return request


def _use_ensemble_member(input_obj, number: int) -> None:
    """Make a prognostic input retrieve the initial conditions of ensemble member ``number`` from ENS.

    Only the inputs of the (fresh, per call) runner are modified, nothing shared:

    - ``step`` is dropped from the configured request keywords, since anemoi would otherwise shift the dates by it
      while the member request is always ``step=0``.
    - The input's request patch is wrapped so that the member override is applied last. It therefore wins over
      anything coming from the checkpoint, the input configuration (``stream``, ``type``, ``number``, ``step``)
      and the configured ``patches``.

    ``MarsInput`` inside a ``Cutout`` (at any depth) are all switched. Any other input cannot take a member, so
    it is left alone with a warning.
    """
    from anemoi.inference.inputs.cutout import Cutout
    from anemoi.inference.inputs.empty import EmptyInput
    from anemoi.inference.inputs.mars import MarsInput

    if isinstance(input_obj, MarsInput):
        configured_patch = input_obj.patch

        def patch(request: dict) -> dict:
            return _member_request(configured_patch(request), number)

        input_obj.kwargs = {k: v for k, v in input_obj.kwargs.items() if k != "step"}
        input_obj.patch = patch  # type: ignore[method-assign]
    elif isinstance(input_obj, Cutout):
        for source in input_obj.sources.values():
            _use_ensemble_member(source, number)
    elif not isinstance(input_obj, EmptyInput):
        LOG.warning(
            f"Ensemble member specification provided but input {input_obj} is not a MarsInput, unable to apply ensemble member specification"
        )


def _get_initial_conditions(config: dict, date: DATE, number: int | None = None) -> dict[str, State]:
    """Get initial conditions for the model.

    With an ensemble member ``number``, the prognostic initial conditions are the member's own ENS fields:
    ``stream=enfo, type=cf, step=0`` for member 0 (no ``number``) and ``stream=enfo, type=pf, number=<number>,
    step=0`` otherwise. Constant and dynamic forcings are not changed. With ``number=None`` nothing is changed.
    """
    # The runner, and so every input, is created fresh in each call: the member override never leaks to other calls
    runner = CascadeRunner(**config)

    states = {}

    # TODO: Replace with a prefetch of all data in the case of dynamics and model uses GribInput during run
    # Use pipes to read and write
    for key in runner.dataset_names:
        dates = [to_datetime(date) + h for h in runner.checkpoint.multi_dataset_metadata[key].lagged]

        if number is not None:
            _use_ensemble_member(runner.prognostics_inputs[key], number)

        states[key] = runner._combine_states(
            runner.prognostics_inputs[key].create_input_state(dates=dates),
            runner.constant_forcings_inputs[key].create_input_state(dates=dates),
            runner.dynamic_forcings_inputs[key].create_input_state(dates=dates),
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
