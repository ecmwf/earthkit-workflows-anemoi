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
from anemoicascade.anemoi_runners import MarsInput, FileInput
from anemoicascade.backends.fieldlist import make_field_list

INPUT_TYPES = Literal["mars", "file"]
inputs = {
    "mars": MarsInput,
    "file": FileInput,
}


def _get_runner_and_input(input_type: INPUT_TYPES, checkpoint, dates: list[dt.datetime], **kwargs):
    runner = DefaultRunner(checkpoint)
    input_cls = inputs[input_type]

    # Dates required in strange format
    f: Callable[[dt.datetime], tuple[int, int]] = lambda d: (
        int(d.strftime("%Y%m%d")),
        d.hour,
    )
    return runner, input_cls(runner.checkpoint, list(map(f, dates)), **kwargs)


def _get_coords(runner: DefaultRunner, target_step: int) -> dict[Literal["param", "step"]]:
    even_steps = (target_step // runner.checkpoint.hour_steps) * runner.checkpoint.hour_steps
    return {
        "param": [
            *runner.checkpoint.prognostic_params,
            *runner.checkpoint.diagnostic_params,
        ],
        "step": np.arange(0, even_steps, runner.checkpoint.hour_steps) + runner.checkpoint.hour_steps,
    }


def _run_model(input_type, ckpt, dates, target_step: int):
    """Run `anemoi.inference` and retrieve data"""

    runner, input_class = _get_runner_and_input(input_type, ckpt, dates)

    coords = _get_coords(runner, target_step)
    hour_steps = runner.checkpoint.hour_steps

    payloads = np.empty((len(coords["param"]), target_step // hour_steps), dtype=object)

    def output_callback(*args, **kwargs):
        # example1 kwargs: {'template': GribField(tcw,None,20240815,1200,0,0), 'step': 240, 'check_nans': True}
        # example2 kwargs: {'stepType': 'accum', 'template': GribField(2t,None,20240815,1200,0,0), 'startStep': 0, 'endStep': 240, 'param': 'tp', 'check_nans': True}
        # example template metadata: {'param': '2t', 'levelist': None, 'validityDate': 20240815, 'validityTime': 1200, 'valid_datetime': '2024-08-15T12:00:00'}
        # args is a tuple with args[0] being numpy data
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
            payloads[coords["param"].index(lookup), (step // hour_steps) - 1] = lambda: value

    runner.run(
        input_fields=input_class.all_fields,
        lead_time=target_step,
        start_datetime=None,  # will be inferred from the input fields
        device="cuda",
        output_callback=output_callback,
        autocast="16",
        progress_callback=tqdm.tqdm,
    )

    return payloads


def _combine_payloads(payloads: np.ndarray[object]) -> ekd.sources.array_list.ArrayFieldList:
    """Combine payloads together into larger ArrayFieldList"""
    complete_data: Optional[ekd.sources.array_list.ArrayFieldList] = None
    for payload in payloads.flatten():
        if payload is None:
            continue

        if complete_data is None:
            complete_data = payload()
        else:
            complete_data = complete_data + payload()

    return complete_data


def _get_dates(start_date: str, input_offset: int, num_inputs: int) -> list[dt.datetime]:
    """Get dates needed for input of model"""
    start_date = dt.datetime.strptime(start_date, "%Y-%m-%dT%H:%M")
    dates = [start_date - dt.timedelta(hours=input_offset * i) for i in range(num_inputs)]
    dates.reverse()
    return dates


def _expand(source: fluent.Action, coords: dict, order: list[str] | None = None):
    """Expand action upon coordinates"""
    if order is not None:
        ordered_dict = OrderedDict()
        for key in order:
            ordered_dict[key] = coords[key]
        return _expand(source, ordered_dict)

    for key, value in coords.items():
        source = source.expand(key, values=value)
    return source


def _underlying_model_runner(
    ckpt: str,
    start_date: str,
    target_step: int = 240,
    *,
    input_offset: int = 6,
    num_inputs: int = 1,
    input_type: INPUT_TYPES = "mars",
    to_xarray: bool = True,
) -> fluent.Payload:
    runner = DefaultRunner(ckpt)
    print(runner.checkpoint.to_dict())
    coords = _get_coords(runner, target_step)

    dates = _get_dates(start_date, input_offset, num_inputs)

    def delayed_prediction(input_type, ckpt, dates, target_step):
        # Delayed prediction

        # Combine payloads
        prediction = _combine_payloads(_run_model(input_type, ckpt, dates, target_step))

        # Convert to xarray
        if to_xarray:
            prediction = prediction.to_xarray(
                variable_key="par_lev_type",
                remapping={"par_lev_type": "{param}_{levelist}"},
            )
            prediction = prediction[coords["param"]].to_dataarray("param")

        return prediction

    model_prediction = fluent.Payload(delayed_prediction, (input_type, ckpt, dates, target_step))
    return model_prediction


def from_model(
    ckpt: str,
    start_date: str,
    target_step: int = 240,
    *,
    input_offset: int = 6,
    num_inputs: int = 1,
    input_type: INPUT_TYPES = "mars",
    expand: bool = False,
    action: fluent.Action = fluent.Action,
    **kwargs,
):
    runner = DefaultRunner(ckpt)
    coords = _get_coords(runner, target_step)

    model_prediction = _underlying_model_runner(
        ckpt,
        start_date,
        target_step,
        input_offset=input_offset,
        num_inputs=num_inputs,
        input_type=input_type,
        **kwargs,
    )
    source = fluent.from_source([model_prediction], action=action)
    if not expand:
        return source
    return _expand(source, coords)


def from_ensemble(
    ckpt: str,
    start_date: str,
    target_step: int = 240,
    ensemble_size: int = 1,
    *,
    expand: bool = False,
    action: fluent.Action = fluent.Action,
    **kwargs,
):
    runner = DefaultRunner(ckpt)
    coords = _get_coords(runner, target_step)

    source = fluent.from_source(
        [_underlying_model_runner(ckpt, start_date, target_step=target_step, **kwargs) for i in range(ensemble_size)],
        coords={"ensemble_member": range(ensemble_size)},
        action=action,
    )

    if not expand:
        return source
    return _expand(source, coords)
