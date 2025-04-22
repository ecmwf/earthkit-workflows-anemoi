# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Fluent API for anemoi inference.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence
from typing import Union

from anemoi.inference.config.run import RunConfiguration
from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from earthkit.workflows import fluent

from anemoi.cascade.inference import run_as_earthkit_from_config
from anemoi.cascade.runner import CascadeRunner

if TYPE_CHECKING:
    from anemoi.inference.input import Input

VALID_CKPT = Union[os.PathLike, str, dict[str, Any]]
ENSEMBLE_MEMBER_SPECIFICATION = Union[int, Sequence[int]]
ENSEMBLE_DIMENSION_NAME: str = "ensemble_member"

LOG = logging.getLogger(__name__)


def _parse_date(date: str | tuple[int, int, int] | datetime.datetime) -> datetime.datetime:
    """Parse date from string or tuple"""
    if isinstance(date, datetime.datetime):
        return date
    elif isinstance(date, str):
        return datetime.datetime.fromisoformat(date)
    else:
        return datetime.datetime(*date)


def _get_initial_conditions(input: Input, date: str | tuple[int, int, int]) -> Any:
    """Get initial conditions for the model"""
    input_state = input.create_input_state(date=_parse_date(date))
    assert isinstance(input_state, dict), "Input state must be a dictionary"
    input_state.pop("_grib_templates_for_output", None)

    return input_state


def _get_initial_conditions_from_config(config: dict[str, Any], date: str | tuple[int, int, int]) -> Any:
    """Get initial conditions for the model"""
    runner = CascadeRunner(config)
    input = runner.create_input()
    return _get_initial_conditions(input, date)


def _get_initial_conditions_ens(input: Input, date: str | tuple[int, int, int], ens_mem: int) -> Any:
    """Get initial conditions for the model"""
    from anemoi.inference.inputs.mars import MarsInput

    if isinstance(input, MarsInput):  # type: ignore
        input.kwargs["number"] = ens_mem  # type: ignore

    input_state = input.create_input_state(date=_parse_date(date))
    assert isinstance(input_state, dict), "Input state must be a dictionary"
    input_state["ensemble_member"] = ens_mem
    input_state.pop("_grib_templates_for_output", None)

    return input_state


def _get_initial_conditions_ens_from_config(
    config: dict[str, Any], date: str | tuple[int, int, int], ens_mem: int
) -> Any:
    """Get initial conditions for the model"""
    runner = CascadeRunner(config)
    input = runner.create_input()
    return _get_initial_conditions_ens(input, date, ens_mem)


def _transform_fake(act: fluent.Action, ens_num: int):
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
    input: Input,
    config: RunConfiguration,
    date: str | tuple[int, int, int],
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION = 1,
    *,
    initial_condition_perturbation: bool = False,
) -> fluent.Action:
    """
    Get the initial conditions for the model

    Parameters
    ----------
    input : Input
        Input object
    config : RunConfiguration
        Configuration object
    date : str | tuple[int, int, int]
        Date to get initial conditions for
    ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION, optional
        Number of ensemble members to get, by default 1
    initial_condition_perturbation : bool, optional
        Whether to get perturbed initial conditions, by default False
        If False, only one initial condition is returned, and
        the ensemble members are simulated by wrapping the action.

    Returns
    -------
    fluent.Action
        Fluent action of the initial conditions
    """
    ensemble_members = _parse_ensemble_members(ensemble_members)
    if initial_condition_perturbation:
        return fluent.from_source(
            [
                [
                    # fluent.Payload(_get_initial_conditions_ens, kwargs=dict(input=input, date=date, ens_mem=ens_mem))
                    fluent.Payload(
                        _get_initial_conditions_ens_from_config, kwargs=dict(config=config, date=date, ens_mem=ens_mem)
                    )
                    for ens_mem in ensemble_members
                ],
            ],  # type: ignore
            coords={"date": [_parse_date(date)], ENSEMBLE_DIMENSION_NAME: ensemble_members},
        )

    # init_condition = fluent.Payload(_get_initial_conditions, kwargs=dict(input=input, date=date))
    init_condition = fluent.Payload(_get_initial_conditions_from_config, kwargs=dict(config=config, date=date))
    single_init = fluent.from_source(
        [
            init_condition,
        ],  # type: ignore
        coords={"date": [_parse_date(date)]},
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


def _time_range(start, end, step):
    """Get a range of timedeltas"""
    while start < end:
        yield start
        start += step


def _expand(runner: CascadeRunner, model_results: fluent.Action, remap: bool = False) -> fluent.Action:
    """Expand model results to the correct dimensions"""

    # Expand by variable
    variables = [*runner.checkpoint.diagnostic_variables, *runner.checkpoint.prognostic_variables]

    # Seperate surface and pressure variables
    surface_vars = [var for var in variables if "_" not in var]
    # pressure_vars_complete = [var for var in variables if "_" in var]

    pressure_vars = list(set(var.split("_")[0] for var in variables if "_" in var))
    # pressure_levels = list(set(var.split('_')[1] for var in variables if "_" in var))

    surface_expansion = model_results.expand(
        ("param", surface_vars), ("param", surface_vars), backend_kwargs=dict(method="sel")
    )
    # pressure_expansion = model_results.expand(('param', pressure_vars_complete), ('param', pressure_vars_complete), backend_kwargs=dict(method="sel", remapping = {'param':"{param}_{level}" if remap else None}))
    pressure_expansion = model_results.expand(
        ("param", pressure_vars), ("param", pressure_vars), backend_kwargs=dict(method="sel")
    )
    # pressure_expansion = pressure_expansion.expand(('level', pressure_levels), ('level', pressure_levels), backend_kwargs=dict(method="sel"))

    return surface_expansion.join(pressure_expansion, dim="param")


def _run_model(
    runner: CascadeRunner, config: RunConfiguration, input_state_source: fluent.Action, lead_time: Any, **kwargs
) -> fluent.Action:
    """
    Run the model, expanding the results to the correct dimensions

    Parameters
    ----------
    runner : Runner
        `anemoi.inference` runner
    config : RunConfiguration
        Configuration object
    input_state_source : fluent.Action
        Fluent action of initial conditions
    lead_time : Any
        Lead time to run out to. Can be a string,
        i.e. `1H`, `1D`, int, or a datetime.timedelta
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
    )

    model_step = runner.checkpoint.timestep
    steps = list(
        map(lambda x: frequency_to_seconds(x) // 3600, _time_range(model_step, lead_time + model_step, model_step))
    )

    model_results = input_state_source.map(model_payload, yields=("step", steps))

    return _expand(runner, model_results, lead_time)


def from_config(
    config: os.PathLike | dict[str, Any],
    overrides: dict[str, Any] | None = None,
    *,
    date: str | tuple[int, int, int] | None = None,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION = 1,
    **kwargs,
) -> fluent.Action:
    """
    Run an anemoi model from a configuration file

    Parameters
    ----------
    config : os.PathLike | dict[str, Any]
        Path to the configuration file, or dictionary of configuration
    overrides : dict[str, Any], optional
        Override for the config, by default None
    date : str | tuple[int, int, int], optional
        Specific override for date, by default None
    ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION , optional
        Number of ensemble members to run, by default 1
    kwargs : dict
        Additional arguments to pass to the runner

    Returns
    -------
    fluent.Action
        Cascade action of the model results

    Examples
    --------
    >>> from anemoi.cascade.fluent import from_config
    >>> from_config("config.yaml", date = "2021-01-01T00:00:00")
    """

    kwargs.update(overrides or {})

    if isinstance(config, os.PathLike):
        override_values = [f"{key}={value}" for key, value in kwargs.items()]
        configuration = RunConfiguration.load(str(config), override_values)
    else:
        config.update(kwargs)
        configuration = RunConfiguration(**config)

    runner = CascadeRunner(configuration)
    runner.checkpoint.validate_environment(on_difference="warn")

    input = runner.create_input()
    input_state_source = get_initial_conditions_source(
        input=input,
        date=date or configuration.date,
        ensemble_members=ensemble_members,
        config=configuration,  # type: ignore
    )

    return _run_model(runner, configuration, input_state_source, configuration.lead_time)


def from_input(
    ckpt: VALID_CKPT,
    input: str | dict[str, Any],  # type: ignore
    date: str | tuple[int, int, int],
    lead_time: Any,
    *,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION = 1,
    **kwargs,
) -> fluent.Action:
    """
    Run an anemoi model from a given input source

    Parameters
    ----------
    ckpt : VALID_CKPT
        Checkpoint to load
    input : str | dict[str, Any]
        `anemoi.inference` input source.
        Can be `mars`, `grib`, etc or a dictionary of input configuration
    date : str | tuple[int, int, int]
        Date to get initial conditions for
    lead_time : Any
        Lead time to run out to. Can be a string,
        i.e. `1H`, `1D`, int, or a datetime.timedelta
    ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION, optional
        Number of ensemble members to run, by default 1
    kwargs : dict
        Additional arguments to pass to the runner

    Returns
    -------
    fluent.Action
        Cascade action of the model results

    Examples
    -------
    >>> from anemoi.cascade.fluent import from_input
    >>> from_input("anemoi_model.ckpt", "mars", date = "2021-01-01T00:00:00", lead_time = "10D")
    """
    from anemoi.inference.config.run import RunConfiguration

    from .runner import CascadeRunner

    config = RunConfiguration(checkpoint=str(ckpt), input=input, **kwargs)

    runner = CascadeRunner(config)
    runner.checkpoint.validate_environment(on_difference="warn")

    input: Input = runner.create_input()
    input_state_source = get_initial_conditions_source(
        input=input, date=date, ensemble_members=ensemble_members, config=config
    )

    return _run_model(runner, config, input_state_source, lead_time)


def from_initial_conditions(
    ckpt: VALID_CKPT,
    initial_conditions: dict[str, Any] | fluent.Action | fluent.Payload | Callable,
    lead_time: Any,
    configuration_kwargs: dict[str, Any] | None = None,
    *,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION | None = None,
    **kwargs,
) -> fluent.Action:
    """
    Run an anemoi model from initial conditions

    Parameters
    ----------
    ckpt : VALID_CKPT
        Checkpoint to load
    initial_conditions : FieldList | fluent.Action | fluent.Payload | Callable
        Initial conditions for the model
        Can be other fluent actions, payloads, or a callable, or a FieldList
        If a fluent action and multiple ensemble member initial conditions
        are included, the dimension must be named `ensemble_member`.
    lead_time : Any
        Lead time to run out to. Can be a string,
        i.e. `1H`, `1D`, int, or a datetime.timedelta
    configuration_kwargs: dict[str, Any]:
        kwargs for `anemoi.inference` configuration
    ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION | None, optional
        Number of ensemble members to run,
        If initial_conditions is a fluent action, with
        multiple ensemble members, this argument can be set to None,
        and the number of ensemble members will be inferred from the action.
        If not set, the number of ensemble members will default to 1.
        by default None.
    kwargs : dict
        Additional arguments to pass to the runner

    Returns
    -------
    fluent.Action
        Cascade action of the model results

    Examples
    --------
    >>> from anemoi.cascade.fluent import from_initial_conditions
    >>> from_initial_conditions("anemoi_model.ckpt", init_conditions, lead_time = "10D")
    """

    config = RunConfiguration(checkpoint=str(ckpt), **(configuration_kwargs or {}))
    runner = CascadeRunner(config, **kwargs)

    runner.checkpoint.validate_environment(on_difference="warn")

    if isinstance(initial_conditions, fluent.Action):
        initial_conditions = initial_conditions
    elif isinstance(initial_conditions, (Callable, fluent.Payload)):
        initial_conditions = fluent.from_source([initial_conditions])  # type: ignore
    else:
        initial_conditions = fluent.from_source([fluent.Payload(lambda: initial_conditions)])  # type: ignore

    if ENSEMBLE_DIMENSION_NAME in initial_conditions.nodes.dims:
        if ensemble_members is None:
            ensemble_members = len(initial_conditions.nodes.coords[ENSEMBLE_DIMENSION_NAME])

        ensemble_members = _parse_ensemble_members(ensemble_members)

        if not len(initial_conditions.nodes.coords[ENSEMBLE_DIMENSION_NAME]) == len(ensemble_members):
            raise ValueError("Number of ensemble members in initial conditions must match `ensemble_members` argument")
        ens_initial_conditions = initial_conditions

    else:
        ens_initial_conditions = initial_conditions.transform(
            _transform_fake,
            list(zip(_parse_ensemble_members(ensemble_members))),  # type: ignore
            (ENSEMBLE_DIMENSION_NAME, ensemble_members),  # type: ignore
        )
    return _run_model(runner, config, ens_initial_conditions, lead_time)


class Action(fluent.Action):

    def infer(
        self, ckpt: VALID_CKPT, lead_time: Any, configuration_kwargs: dict[str, Any] | None = None, **kwargs
    ) -> fluent.Action:
        """
        Map a model prediction to all nodes within
        the graph, using them as initial conditions

        Parameters
        ----------
        ckpt : VALID_CKPT
            Checkpoint to load
        lead_time : Any
            Lead time to run out to. Can be a string,
            i.e. `1H`, `1D`, int, or a datetime.timedelta
        configuration_kwargs: dict[str, Any]:
            kwargs for anemoi.inference configuration
        kwargs : dict
            Additional arguments to pass to the runner


        Returns
        -------
        fluent.Action
            Cascade action of the model results
        """
        return from_initial_conditions(ckpt, self, lead_time, configuration_kwargs=configuration_kwargs, **kwargs)


fluent.Action.register("anemoi", Action)
