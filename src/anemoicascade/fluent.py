from collections import OrderedDict
import numpy as np
import datetime as dt
from typing import Any, Callable, Literal, Optional

from cascade import fluent

import earthkit.data as ekd
import tqdm

from anemoi.inference.runner import Runner, DefaultRunner
from anemoicascade.anemoi_runners import MarsInput, FileInput, RequestBasedInput
from anemoicascade.backends.fieldlist import make_field_list

INPUT_TYPES = Literal["mars", "file"]
INPUTS: dict[str, RequestBasedInput] = {
    "mars": MarsInput,
    "file": FileInput,
}



def _retrieve_initial_conditions(
    input_type: INPUT_TYPES,
    checkpoint: str,
    start_date: str,
    ensemble_number: int | None = None,
) -> ekd.sources.array_list.ArrayFieldList:
    """
    Retrieve initial conditions for the model

    Parameters
    ----------
    input_type : INPUT_TYPES
        Source of the initial conditions
    checkpoint : str
        Checkpoint of the model to use
    start_date : str
        Start date of the initial conditions
    ensemble_number : int | None, optional
        Id number of the ensemble, by default None

    Returns
    -------
    ekd.sources.array_list.ArrayFieldList
        Initial conditions for the model for the given ensemble member
    """
    runner = DefaultRunner(checkpoint)
    # Dates required in strange format
    f: Callable[[dt.datetime], tuple[int, int]] = lambda d: (
        int(d.strftime("%Y%m%d")),
        d.hour,
    )
    start_date = dt.datetime.strptime(start_date, "%Y-%m-%dT%H:%M")

    dates = [start_date + dt.timedelta(hours=x) for x in runner.lagged]
    input_cls = INPUTS[input_type](runner.checkpoint, list(map(f, dates)))
    return input_cls.all_fields


def _get_coords(runner: DefaultRunner, lead_time: int) -> dict[Literal["param", "step"]]:
    """
    Get coordinates for the model output from the input
    """
    even_steps = (lead_time // runner.checkpoint.hour_steps) * runner.checkpoint.hour_steps
    return {
        "param": [
            *runner.checkpoint.prognostic_params,
            *runner.checkpoint.diagnostic_params,
        ],
        "step": [x + runner.checkpoint.hour_steps for x in range(0, even_steps, runner.checkpoint.hour_steps)],
    }


def _run_model(initial_conditions, ckpt, lead_time: int, **kwargs):
    """
    Underlying function to run the model

    Parameters
    ----------
    initial_conditions :
        Initial conditions for the model
    ckpt :
        Checkpoint pointing to the model to load
    lead_time : int
        Number of steps to run the model for

    Returns
    -------
    ekd.sources.array_list.ArrayFieldList:
        Prediction from the model combined into one field list
    """    
    runner = DefaultRunner(ckpt)
    coords = _get_coords(runner, lead_time)
    hour_steps = runner.checkpoint.hour_steps

    payloads = np.empty((len(coords["param"]), lead_time // hour_steps), dtype=object)

    def output_callback(*args, **kwargs):
        if "step" in kwargs or "endStep" in kwargs:
            data = args[0]
            template = kwargs.pop("template")

            param = kwargs.get("param", template._metadata.get("param", ""))

            level = template._metadata.get("levelist", 0)
            level = level if level else 0  # getting around None

            lookup = f"{param}_{level}" if level > 0 else param
            if lookup not in coords["param"]:  # Check if given value is expected and ignore otherwise
                return

            step = kwargs.get("step") if "step" in kwargs else kwargs.get("endStep")
            value = make_field_list(data, template, **kwargs)
            payloads[coords["param"].index(lookup), (step // hour_steps) - 1] = value
    
    device = kwargs.pop('device', None)
    run_dict = dict(
        device=device or "cuda",
        autocast="16",
        progress_callback=tqdm.tqdm,
    )
    run_dict.update(kwargs)

    runner.run(
        input_fields=initial_conditions,
        lead_time=lead_time,
        start_datetime=None,  # will be inferred from the input fields
        output_callback=output_callback,
        **run_dict
    )

    complete_data: ekd.sources.array_list.ArrayFieldList = None
    for payload in payloads.flatten():
        if payload is None:
            continue

        if complete_data is None:
            complete_data = payload
        else:
            complete_data = complete_data + payload

    return complete_data


def _expand(source: fluent.Action, coords: dict, order: list[str] | None = None):
    """Expand action upon coordinates"""
    if order is not None:
        ordered_dict = OrderedDict()
        for key in order:
            ordered_dict[key] = coords[key]
        return _expand(source, ordered_dict)

    for key, value in coords.items():
        dataarray = (key, value)
        source = source.expand(dim=dataarray, internal_dim=dataarray, backend_kwargs=dict(method="sel", remapping = {'param':"{param}_{level}"}))
    return source


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
    if devices is not None:
        if isinstance(devices, str):
            devices = [devices] * num_ensembles
        assert len(devices) == num_ensembles, "Number of devices should match the number of ensembles"

    source = fluent.from_source(
        [
            fluent.Payload(
                _retrieve_initial_conditions,
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
            _run_model,
            kwargs=dict(ckpt = ckpt, lead_time = lead_time, device=devices[ensemble_number] if devices is not None else 'None', **kwargs),
        )
        models.append(model)

    prediction = source.map(models)

    return _expand(prediction, _get_coords(DefaultRunner(ckpt), lead_time))

