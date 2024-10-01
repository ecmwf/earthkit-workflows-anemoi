
import functools
from typing import Any, Callable, Literal

from cascade import fluent


from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.runner import DefaultRunner

from anemoicascade.inference import run_model, retrieve_initial_conditions, get_coords, INPUT_TYPES
import numpy as np
import xarray as xr


def _expand(source: fluent.Action, coords: dict[Literal["param", "step"]]):
    """Expand action upon coordinates"""

    source = source.expand(('step', coords['step']), ('step', coords['step']), backend_kwargs=dict(method="sel"))

    surface_vars  = [var for var in coords["param"] if "_" not in var]
    pressure_vars = [var for var in coords["param"] if "_" in var]

    surface_expansion = source.expand(('param', surface_vars), ('param', surface_vars), backend_kwargs=dict(method="sel"))
    pressure_vars = source.expand(('param', pressure_vars), ('param', pressure_vars), backend_kwargs=dict(method="sel", remapping = {'param':"{param}_{level}"}))
    
    return surface_expansion.join(pressure_vars, dim = 'param')

def from_model(
    ckpt,
    start_date: str,
    lead_time: int,
    *,
    num_ensembles: int = 1,
    input_type: INPUT_TYPES = "mars",
    action: type[fluent.Action] = fluent.Action,
    devices: list[str] | str | None = None,
    input_kwargs: dict[str, Any] = None,
    **kwargs,
) -> fluent.Action:
    """
    Create a Cascade Graph from a model prediction
    
    Will be automatically expanded to the correct dimensions
    of param, step and ensemble.

    Parameters
    ----------
    ckpt :
        Location of ckpt to load
    start_date : str
        Start date of prediction, used to get initial conditions
    lead_time : int
        Hours to predict out to
    num_ensembles : int, optional
        Number of ensembles to create, by default 1
    input_type : INPUT_TYPES, optional
        Source of input data, by default "mars"
    action : type[fluent.Action], optional
        Cascade action to use, by default fluent.Action
    devices : list[str] | str | None, optional
        Device assignment of ensemble members, must have length == num_ensembles, by default None
    input_kwargs : dict[str, Any], optional
        Kwargs to pass to initial condition retrieval, by default None

    Returns
    -------
    fluent.Action
        Cascade Action of the model prediction
    """    
    Checkpoint(ckpt).validate_environment(on_difference='warn')  # Check if checkpoint is valid

    if devices is not None:
        if isinstance(devices, str):
            devices = [devices] * num_ensembles
        assert len(devices) == num_ensembles, "Number of devices should match the number of ensembles"
    

    source = fluent.from_source(
        [
            fluent.Payload(
                retrieve_initial_conditions,
                (input_type, ckpt, start_date, ensemble_number),
                kwargs=input_kwargs or {},
            )
            for ensemble_number in range(num_ensembles)
        ],
        coords={"ensemble_member": range(num_ensembles)},
        action=action,
    )
    models = []
    for ensemble_number in range(num_ensembles):
        model = fluent.Payload(
            run_model,
            kwargs=dict(ckpt = ckpt, lead_time = lead_time, device=devices[ensemble_number] if devices is not None else None, **kwargs),
        )
        models.append(model)

    prediction = source.map(models)

    return _expand(prediction, get_coords(DefaultRunner(ckpt), lead_time))

def with_initial_conditions(
    initial_conditions,
    ckpt: fluent.Action | fluent.Payload | Callable | Any,
    lead_time: int,
    *,
    action: type[fluent.Action] = None,
    devices: list[str] | str | None = None,
    **kwargs,
    ):
    """
    _summary_

    Parameters
    ----------
    initial_conditions : _type_
        _description_
    ckpt : _type_
        _description_
    lead_time : int
        _description_
    action : type[fluent.Action], optional
        _description_, by default None
    devices : list[str] | str | None, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """    
    if isinstance(initial_conditions, fluent.Action):
        initial_conditions = initial_conditions
    elif isinstance(initial_conditions, (Callable, fluent.Payload)):
        initial_conditions = fluent.from_source([initial_conditions], action=action or fluent.Action)
    else:
        initial_conditions = fluent.from_source([fluent.Payload(lambda: initial_conditions)], action=action or fluent.Action)

    models = fluent.Payload(
        run_model,
        kwargs=dict(ckpt = ckpt, lead_time = lead_time, devices=devices, **kwargs),
    )
    prediction = initial_conditions.map(models)
    return _expand(prediction, get_coords(DefaultRunner(ckpt), lead_time))
     

class AnemoiActions(fluent.Action):
    def infer(self, ckpt: str, lead_time: int, **kwargs) -> fluent.Action:
        """
        Map a model prediction to all nodes within 
        the graph, using them as initial conditions

        Parameters
        ----------
        ckpt : str
            Model checkpoint to load
        lead_time : int
            Lead time to predict out to in hours

        Returns
        -------
        fluent.Action
            Expanded action with model predictions
        """        
        return with_initial_conditions(self, ckpt, lead_time, **kwargs)
        
    
    
fluent.Action.register("anemoi", AnemoiActions)