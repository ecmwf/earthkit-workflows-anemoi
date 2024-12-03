from __future__ import annotations

import os
from typing import Any, Callable, TYPE_CHECKING
import datetime

from cascade import fluent

import logging

from anemoicascade.inference import run_as_earthkit, collect_as_earthkit
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.dates import frequency_to_seconds


if TYPE_CHECKING:
    from anemoi.inference.input import Input
    from anemoi.inference.runner import Runner

LOG = logging.getLogger(__name__)

def _parse_date(date: str | tuple[int, int, int] | datetime.datetime) -> datetime.datetime:
    """Parse date from string or tuple"""
    if isinstance(date, datetime.datetime):
        return date
    elif isinstance(date, str):
        return datetime.datetime.fromisoformat(date)
    else:
        return datetime.datetime(*date)

def _get_initial_conditions(input: Input, date: str | tuple[int, int, int], ens_mem: int = 0) -> Any:
    """Get initial conditions for the model"""
    input_state = input.create_input_state(date = _parse_date(date))
    input_state.pop('_grib_templates_for_output', None)
    return input_state

def get_initial_conditions_source(input: Input, date: str | tuple[int, int, int], ensemble_members: int = 1) -> fluent.Action:
    # init_condition = 
    return fluent.from_source(
        [
            [fluent.Payload(_get_initial_conditions, kwargs=dict(input = input, date = date, ens_mem = i)) for i in range(ensemble_members)],
        ],
        coords = {'date': [date], "member": range(ensemble_members)},
    )

def _time_range(start, end, step):
    """Get a range of timedeltas"""
    while start < end:
        yield start
        start += step

def _expand(runner: Runner, model_results: fluent.Action, lead_time: datetime.timedelta, remap: bool = False) -> fluent.Action:
    """Expand model results to the correct dimensions"""
    model_step = runner.checkpoint.timestep
    steps = list(map(lambda x: frequency_to_seconds(x) // 3600, _time_range(model_step, lead_time + model_step, model_step)))

    # Expand by time
    model_results_by_time: fluent.Action = model_results.expand(('step', steps), ('step', steps), backend_kwargs=dict(method="sel"))

    # Expand by variable
    variables = [*runner.checkpoint.diagnostic_variables, *runner.checkpoint.prognostic_variables]

    # Seperate surface and pressure variables
    surface_vars  = [var for var in variables if "_" not in var]
    # pressure_vars = [var for var in variables if "_" in var]
    pressure_vars = list(set(var.split('_')[0] for var in variables if "_" in var))

    surface_expansion = model_results_by_time.expand(('param', surface_vars), ('param', surface_vars), backend_kwargs=dict(method="sel"))
    # pressure_vars = model_results_by_time.expand(('param', pressure_vars), ('param', pressure_vars), backend_kwargs=dict(method="sel", remapping = {'param':"{param}_{level}" if remap else None}))
    pressure_vars = model_results_by_time.expand(('param', pressure_vars), ('param', pressure_vars), backend_kwargs=dict(method="sel"))

    return surface_expansion.join(pressure_vars, dim = 'param')

    # model_results_by_variable: fluent.Action = model_results_by_time.expand(('param', variables), ('param', variables), backend_kwargs=dict(method="sel", remapping = {'param':"{param}_{level}"} if remap else None))
    # return model_results_by_variable

def _run_model(runner: Runner, input_state_source: fluent.Action, lead_time: Any, **kwargs) -> fluent.Action:
    """
    Run the model, expanding the results to the correct dimensions

    Parameters
    ----------
    runner : Runner
        `anemoi.inference` runner
    input_state_source : fluent.Action
        Fluent action of initial conditions
    lead_time : Any
        Lead time to run out to. Can be a string, 
        i.e. `1H`, `1D`, int, or a datetime.timedelta

    Returns
    -------
    fluent.Action
        Cascade action of the model results
    """
    lead_time = to_timedelta(lead_time)

    model_payload = fluent.Payload(collect_as_earthkit, kwargs=dict(runner=runner, lead_time=lead_time, **kwargs))
    model_results = input_state_source.map(model_payload)
    
    return _expand(runner, model_results, lead_time)

def from_config(
    config: os.PathLike,
    overrides: dict[str, Any] = None,
    *,
    date: str | tuple[int, int, int] = None,
    ensemble_members: int = 1,
    **kwargs,
) -> fluent.Action: 
    """
    Run an anemoi model from a configuration file

    Parameters
    ----------
    config : os.PathLike
        Path to the configuration file
    overrides : dict[str, Any], optional
        Override for the config, by default None
    date : str | tuple[int, int, int], optional
        Specific override for date, by default None
    ensemble_members : int, optional
        Number of ensemble members to run, by default 1
    Returns
    -------
    fluent.Action
        Cascade action of the model results

    Examples
    --------
    >>> from anemoicascade import from_config
    >>> from_config("config.yaml", date = "2021-01-01T00:00:00")
    """
    
    from anemoi.inference.config import load_config
    from anemoi.inference.runners.default import DefaultRunner
    
    kwargs.update(overrides or {})
    overrides = [f"{key}={value}" for key, value in kwargs.items()]

    config = load_config(config, overrides)

    runner = DefaultRunner(config)
    runner.checkpoint.validate_environment(on_difference='warn')

    input = runner.create_input()

    input_state_source = get_initial_conditions_source(input = input, date = date or config.date, ensemble_members = ensemble_members)

    return _run_model(runner, input_state_source, config.lead_time)


def from_input(
    ckpt: os.PathLike,
    input: str | dict[str, Any],
    date: str | tuple[int, int, int],
    lead_time: Any,
    *,
    ensemble_members: int = 1,
    **kwargs,
) -> fluent.Action:
    """
    Run an anemoi model from a given input source

    Parameters
    ----------
    ckpt : os.PathLike
        Checkpoint to load
    input : str | dict[str, Any]
        `anemoi.inference` input source.
        Can be `mars`, `grib`, etc or a dictionary of input configuration
    date : str | tuple[int, int, int]
        Date to get initial conditions for
    lead_time : Any
        Lead time to run out to. Can be a string, 
        i.e. `1H`, `1D`, int, or a datetime.timedelta
    ensemble_members : int, optional
        Number of ensemble members to run, by default 1

    Returns
    -------
    fluent.Action
        Cascade action of the model results

    Examples
    -------
    >>> from anemoicascade import from_input
    >>> from_input("anemoi_model.ckpt", "mars", date = "2021-01-01T00:00:00", lead_time = "10D")
    """    
    from anemoi.inference.runners.default import DefaultRunner
    from anemoi.inference.inputs import create_input
    from anemoi.inference.config import Configuration

    config = Configuration(checkpoint=ckpt, input = input, **kwargs)

    runner = DefaultRunner(config)
    runner.checkpoint.validate_environment(on_difference='warn')

    input = create_input(runner, input)

    input_state_source = get_initial_conditions_source(input = input, date = date, ensemble_members = ensemble_members)

    return _run_model(runner, input_state_source, lead_time)

def from_initial_conditions(
    ckpt: os.PathLike,
    initial_conditions: dict[str, Any] | fluent.Action | fluent.Payload | Callable,
    lead_time: Any,
    **kwargs,
) -> fluent.Action:
    """
    Run an anemoi model from initial conditions

    Parameters
    ----------
    ckpt : os.PathLike
        Checkpoint to load
    initial_conditions : dict[str, Any] | fluent.Action | fluent.Payload | Callable
        Initial conditions for the model
        Can be other fluent actions, payloads, or a callable, or a input state dictionary
    lead_time : Any
        Lead time to run out to. Can be a string, 
        i.e. `1H`, `1D`, int, or a datetime.timedelta

    Returns
    -------
    fluent.Action
        Cascade action of the model results

    Examples
    --------
    >>> from anemoicascade import from_initial_conditions
    >>> from_initial_conditions("anemoi_model.ckpt", init_conditions, lead_time = "10D")
    """
    from anemoi.inference.runners.simple import SimpleRunner
    runner = SimpleRunner(ckpt, **kwargs)
    runner.checkpoint.validate_environment(on_difference='warn')

    if isinstance(initial_conditions, fluent.Action):
        initial_conditions = initial_conditions
    elif isinstance(initial_conditions, (Callable, fluent.Payload)):
        initial_conditions = fluent.from_source([initial_conditions])
    else:
        initial_conditions = fluent.from_source([fluent.Payload(lambda: initial_conditions)])

    return _run_model(runner, initial_conditions, lead_time)


class AnemoiActions(fluent.Action):

    def infer(self, ckpt: os.PathLike, lead_time: Any, **kwargs) -> fluent.Action:
        """
        Map a model prediction to all nodes within 
        the graph, using them as initial conditions

        Parameters
        ----------
        ckpt : os.PathLike
            Checkpoint to load
        lead_time : Any
            Lead time to run out to. Can be a string, 
            i.e. `1H`, `1D`, int, or a datetime.timedelta

        Returns
        -------
        fluent.Action
            Cascade action of the model results
        """        
        return from_initial_conditions(ckpt, self, lead_time, **kwargs)

    
fluent.Action.register("anemoi", AnemoiActions)