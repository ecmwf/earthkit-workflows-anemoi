from __future__ import annotations

import os
from typing import Any, Callable, TYPE_CHECKING, Union
import datetime

from cascade import fluent

import logging

from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.dates import frequency_to_seconds


from anemoicascade.inference import run_as_earthkit, collect_as_earthkit


if TYPE_CHECKING:
    from anemoi.inference.input import Input
    from anemoi.inference.runner import Runner

VALID_CKPT = Union[os.PathLike, str, dict[str, Any]]
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
    input_state = input.create_input_state(date = _parse_date(date))
    input_state.pop('_grib_templates_for_output', None)
    return input_state

def _get_initial_conditions_ens(input: Input, date: str | tuple[int, int, int], ens_mem: int) -> Any:
    """Get initial conditions for the model"""
    from anemoi.inference.inputs.mars import MarsInput
    if isinstance(input, MarsInput):
        input.kwargs['number'] = ens_mem
    input_state = input.create_input_state(date = _parse_date(date))
    input_state.pop('_grib_templates_for_output', None)
    return input_state

def _transform_fake(act: fluent.Action, ens_num: int):
    """Transform the action to simulate ensemble members"""
    def _empty_payload(x, ens_num: int):
        _ = ens_num
        assert isinstance(x, dict), "Input state must be a dictionary"
        return x
    return act.map(fluent.Payload(_empty_payload, [fluent.Node.input_name(0), ens_num]))

def get_initial_conditions_source(input: Input, date: str | tuple[int, int, int], ensemble_members: int = 1, *, initial_condition_perturbation: bool = False) -> fluent.Action:
    """
    Get the initial conditions for the model

    Parameters
    ----------
    input : Input
        Input object
    date : str | tuple[int, int, int]
        Date to get initial conditions for
    ensemble_members : int, optional
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
    if initial_condition_perturbation:
        return fluent.from_source(
            [
                [fluent.Payload(_get_initial_conditions_ens, kwargs=dict(input = input, date = date, ens_mem = ens_mem)) for ens_mem in range(ensemble_members)],
            ],
            coords = {'date': [date], "ensemble_member": range(ensemble_members)},
        )
    
    init_condition = fluent.Payload(_get_initial_conditions, kwargs=dict(input = input, date = date))
    single_init = fluent.from_source(
        [
            init_condition,
        ],
        coords = {'date': [date]},
    )
    # Wrap with empty payload to simulate ensemble members
    return single_init.transform(_transform_fake, list(zip(range(ensemble_members))), 'ensemble_member')

def _time_range(start, end, step):
    """Get a range of timedeltas"""
    while start < end:
        yield start
        start += step

def _expand(runner: Runner, model_results: fluent.Action, remap: bool = False) -> fluent.Action:
    """Expand model results to the correct dimensions"""

    # Expand by variable
    variables = [*runner.checkpoint.diagnostic_variables, *runner.checkpoint.prognostic_variables]

    # Seperate surface and pressure variables
    surface_vars  = [var for var in variables if "_" not in var]
    # pressure_vars_complete = [var for var in variables if "_" in var]

    pressure_vars = list(set(var.split('_')[0] for var in variables if "_" in var))
    # pressure_levels = list(set(var.split('_')[1] for var in variables if "_" in var))

    surface_expansion = model_results.expand(('param', surface_vars), ('param', surface_vars), backend_kwargs=dict(method="sel"))
    # pressure_expansion = model_results.expand(('param', pressure_vars_complete), ('param', pressure_vars_complete), backend_kwargs=dict(method="sel", remapping = {'param':"{param}_{level}" if remap else None}))
    pressure_expansion = model_results.expand(('param', pressure_vars), ('param', pressure_vars), backend_kwargs=dict(method="sel"))
    # pressure_expansion = pressure_expansion.expand(('level', pressure_levels), ('level', pressure_levels), backend_kwargs=dict(method="sel"))

    return surface_expansion.join(pressure_expansion, dim = 'param')

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

    model_payload = fluent.Payload(run_as_earthkit, kwargs=dict(runner=runner, lead_time=lead_time, **kwargs))

    model_step = runner.checkpoint.timestep
    steps = list(map(lambda x: frequency_to_seconds(x) // 3600, _time_range(model_step, lead_time + model_step, model_step)))

    model_results = input_state_source.map(model_payload, yields=('step', steps))
    
    return _expand(runner, model_results, lead_time)

def from_config(
    config: os.PathLike | dict[str, Any],
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
    config : os.PathLike | dict[str, Any]
        Path to the configuration file, or dictionary of configuration
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
    
    from anemoi.inference.config import load_config, Configuration

    from anemoi.inference.runners.default import DefaultRunner
    
    kwargs.update(overrides or {})

    if isinstance(config, os.PathLike):
        overrides = [f"{key}={value}" for key, value in kwargs.items()]
        configuration = load_config(config, overrides)
    else:
        config.update(kwargs)
        configuration = Configuration(**config)

    runner = DefaultRunner(configuration)
    runner.checkpoint.validate_environment(on_difference='warn')

    input = runner.create_input()
    input_state_source = get_initial_conditions_source(input = input, date = date or configuration.date, ensemble_members = ensemble_members)

    return _run_model(runner, input_state_source, configuration.lead_time)


def from_input(
    ckpt: VALID_CKPT,
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

    from anemoi.inference.config import Configuration

    config = Configuration(checkpoint=ckpt, input = input, **kwargs)

    runner = DefaultRunner(config)
    runner.checkpoint.validate_environment(on_difference='warn')

    input = runner.create_input()
    input_state_source = get_initial_conditions_source(input = input, date = date, ensemble_members = ensemble_members)

    return _run_model(runner, input_state_source, lead_time)

def from_initial_conditions(
    ckpt: VALID_CKPT,
    initial_conditions: dict[str, Any] | fluent.Action | fluent.Payload | Callable,
    lead_time: Any,
    configuration_kwargs: dict[str, Any] | None = None,
    *,
    ensemble_members: int = 1,
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
    lead_time : Any
        Lead time to run out to. Can be a string, 
        i.e. `1H`, `1D`, int, or a datetime.timedelta
    configuration_kwargs: dict[str, Any]:
        kwargs for `anemoi.inference` configuration
    ensemble_members : int, optional
        Number of ensemble members to run, by default 1
    Returns
    -------
    fluent.Action
        Cascade action of the model results

    Examples
    --------
    >>> from anemoicascade import from_initial_conditions
    >>> from_initial_conditions("anemoi_model.ckpt", init_conditions, lead_time = "10D")
    """
    from anemoicascade.runner import CascadeRunner

    runner = CascadeRunner.from_kwargs(checkpoint = ckpt, configuration_kwargs = configuration_kwargs, **kwargs)
    runner.checkpoint.validate_environment(on_difference='warn')

    if isinstance(initial_conditions, fluent.Action):
        initial_conditions = initial_conditions
    elif isinstance(initial_conditions, (Callable, fluent.Payload)):
        initial_conditions = fluent.from_source([initial_conditions])
    else:
        initial_conditions = fluent.from_source([fluent.Payload(lambda: initial_conditions)])

    ens_initial_conditions =  initial_conditions.transform(_transform_fake, list(zip(range(ensemble_members))), 'ensemble_member')
    return _run_model(runner, ens_initial_conditions, lead_time)


class AnemoiActions(fluent.Action):

    def infer(self, ckpt: os.PathLike, lead_time: Any, configuration_kwargs: dict[str, Any] | None = None, **kwargs) -> fluent.Action:
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
        configuration_kwargs: dict[str, Any]:
            kwargs for anemoi.inference configuration

        Returns
        -------
        fluent.Action
            Cascade action of the model results
        """        
        return from_initial_conditions(ckpt, self, lead_time, configuration_kwargs = configuration_kwargs, **kwargs)

    
fluent.Action.register("anemoi", AnemoiActions)