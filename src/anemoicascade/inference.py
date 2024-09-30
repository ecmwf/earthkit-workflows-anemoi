import numpy as np
import datetime as dt
from typing import Any, Callable, Literal


import earthkit.data as ekd
import tqdm

from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.runner import Runner, DefaultRunner
from anemoicascade.anemoi_runners import MarsInput, FileInput, RequestBasedInput
from anemoicascade.backends.fieldlist import make_field_list

INPUT_TYPES = Literal["mars", "file"]
INPUTS: dict[str, RequestBasedInput] = {
    "mars": MarsInput,
    "file": FileInput,
}



def retrieve_initial_conditions(
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


def get_coords(runner: DefaultRunner, lead_time: int) -> dict[Literal["param", "step"]]:
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


def run_model(initial_conditions, ckpt, lead_time: int, **kwargs):
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
    runner = DefaultRunner(ckpt, verbose=False)
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
