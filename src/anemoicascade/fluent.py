from collections import OrderedDict
import io
import numpy as np
import xarray as xr
import functools
import datetime as dt
from typing import Callable, Literal, Optional

from cascade import fluent
from cascade import backends

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


def _get_coords(runner: DefaultRunner, target_step: int) -> dict[Literal["param", "step"]]:
    even_steps = (target_step // runner.checkpoint.hour_steps) * runner.checkpoint.hour_steps
    return {
        "param": [
            *runner.checkpoint.prognostic_params,
            *runner.checkpoint.diagnostic_params,
        ],
        "step": [x + runner.checkpoint.hour_steps for x in range(0, even_steps, runner.checkpoint.hour_steps)],
    }


def _run_model(initial_conditions, ckpt, target_step: int, **kwargs):
    runner = DefaultRunner(ckpt)
    coords = _get_coords(runner, target_step)
    hour_steps = runner.checkpoint.hour_steps

    payloads = np.empty((len(coords["param"]), target_step // hour_steps), dtype=object)

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

    runner.run(
        input_fields=initial_conditions,
        lead_time=target_step,
        start_datetime=None,  # will be inferred from the input fields
        device="cuda",
        output_callback=output_callback,
        autocast="16",
        progress_callback=tqdm.tqdm,
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
        source = source.expand(dim=dataarray, internal_dim=dataarray, backend_kwargs=dict(method="sel"))
    return source


def from_model(
    ckpt,
    start_date: str,
    target_step: int,
    *,
    num_ensembles: int = 1,
    input_type: INPUT_TYPES = "mars",
    action: fluent.Action = fluent.Action,
) -> fluent.Action:
    source = fluent.from_source(
        [
            fluent.Payload(
                _retrieve_initial_conditions,
                (input_type, ckpt, start_date, ensemble_number),
            )
            for ensemble_number in range(num_ensembles)
        ],
        coords={"ensemble_member": range(num_ensembles)},
        action=action,
    )

    prediction = source.map(
        [fluent.Payload(_run_model, kwargs = dict(ckpt = ckpt, target_step = target_step)) for _ in range(num_ensembles)],
    )

    return _expand(prediction, _get_coords(DefaultRunner(ckpt), target_step))
