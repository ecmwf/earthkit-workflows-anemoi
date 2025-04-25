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

import functools
import logging
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from anemoi.inference.config.run import RunConfiguration
from earthkit.workflows import fluent

from anemoi.cascade.inference import _transform_fake
from anemoi.cascade.inference import get_initial_conditions_source
from anemoi.cascade.inference import parse_ensemble_members
from anemoi.cascade.inference import run_model
from anemoi.cascade.runner import CascadeRunner
from anemoi.cascade.types import ENSEMBLE_DIMENSION_NAME

if TYPE_CHECKING:
    from anemoi.cascade.types import ENSEMBLE_MEMBER_SPECIFICATION
    from anemoi.cascade.types import VALID_CKPT

LOG = logging.getLogger(__name__)


def from_config(
    config: os.PathLike | dict[str, Any] | RunConfiguration,
    overrides: dict[str, Any] | None = None,
    *,
    date: str | tuple[int, int, int] | None = None,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION = 1,
    **kwargs,
) -> fluent.Action:
    """
    Run an anemoi inference model from a configuration file

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
        earthkit.workflows action of the model results

    Examples
    --------
    >>> from anemoi.cascade.fluent import from_config
    >>> from_config("config.yaml", date = "2021-01-01T00:00:00")
    """

    kwargs.update(overrides or {})

    if isinstance(config, os.PathLike):
        configuration = RunConfiguration.load(str(config), overrides=kwargs)
    elif isinstance(config, dict):
        config.update(kwargs)
        configuration = RunConfiguration(**config)
    elif isinstance(config, RunConfiguration):
        config_dump = config.model_dump()
        config_dump.update(kwargs)
        configuration = RunConfiguration(**config_dump)
    else:
        raise TypeError(f"Invalid type for config: {type(config)}. " "Must be os.PathLike, dict, or RunConfiguration.")

    runner = CascadeRunner(configuration)
    runner.checkpoint.validate_environment(on_difference="warn")

    input_state_source = get_initial_conditions_source(
        config=configuration,  # type: ignore
        date=date or configuration.date,
        ensemble_members=ensemble_members,
    )

    return run_model(runner, configuration, input_state_source, configuration.lead_time)


def from_input(
    ckpt: VALID_CKPT,
    input: str | dict[str, Any],
    date: str | tuple[int, int, int],
    lead_time: Any,
    *,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION = 1,
    **kwargs,
) -> fluent.Action:
    """
    Run an anemoi inference model from a given input source

    Parameters
    ----------
    ckpt : VALID_CKPT
        Checkpoint to load
    input : str | dict[str, Any]
        `anemoi.inference` input source.
        Can be `mars`, `grib`, etc or a dictionary of input configuration,
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
        earthkit.workflows action of the model results

    Examples
    -------
    >>> from anemoi.cascade.fluent import from_input
    >>> from_input("anemoi_model.ckpt", "mars", date = "2021-01-01T00:00:00", lead_time = "10D")
    """
    config = RunConfiguration(checkpoint=str(ckpt), input=input, **kwargs)

    runner = CascadeRunner(config)
    runner.checkpoint.validate_environment(on_difference="warn")

    input_state_source = get_initial_conditions_source(date=date, ensemble_members=ensemble_members, config=config)

    return run_model(runner, config, input_state_source, lead_time)


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
    Run an anemoi inference model from initial conditions

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
        earthkit.workflows action of the model results

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
        initial_conditions = fluent.from_source([fluent.Payload(lambda: initial_conditions)], dims=["date"])  # type: ignore

    if ENSEMBLE_DIMENSION_NAME in initial_conditions.nodes.dims:
        if ensemble_members is None:
            ensemble_members = len(initial_conditions.nodes.coords[ENSEMBLE_DIMENSION_NAME])

        ensemble_members = parse_ensemble_members(ensemble_members)

        if not len(initial_conditions.nodes.coords[ENSEMBLE_DIMENSION_NAME]) == len(ensemble_members):
            raise ValueError("Number of ensemble members in initial conditions must match `ensemble_members` argument")
        ens_initial_conditions = initial_conditions

    else:
        ens_initial_conditions = initial_conditions.transform(
            _transform_fake,
            list(zip(parse_ensemble_members(ensemble_members))),  # type: ignore
            (ENSEMBLE_DIMENSION_NAME, parse_ensemble_members(ensemble_members)),  # type: ignore
        )
    return run_model(runner, config, ens_initial_conditions, lead_time)


def create_dataset(
    config: dict[str, Any] | os.PathLike, path: os.PathLike, *, overwrite: bool = False, test: bool = False
) -> fluent.Action:
    """
    Create an anemoi dataset from a configuration.

    Will load a sample from mars before returning the action.

    Parameters
    ----------
    config : dict[str, Any] | os.PathLike
        Configuration to use
    path : os.PathLike
        Path to save the dataset to
    overwrite : bool, optional
        Whether to overwrite the dataset if it exists, by default False
    test : bool, optional
        Build a small dataset, using only the first dates. And, when possible, using low resolution and less ensemble members,
        by default False

    Returns
    -------
    fluent.Action
        earthkit.workflows action to create the dataset

    Examples
    --------
    >>> from anemoi.cascade.fluent import create_dataset
    >>> create_dataset("dataset_recipe.yaml", "output_dir/dataset.zarr")
    """
    from anemoi.datasets.create import creator_factory

    options = {"config": config, "path": os.path.abspath(path), "overwrite": overwrite, "test": test}

    total = creator_factory("init", **options).run()

    init = fluent.from_source([lambda: 0], dims=["source"])

    def get_options(part: int):
        opt = options.copy()
        opt["parts"] = f"{part+1}/{total}"
        return opt

    def get_task(name: str, opt: dict[str, Any]) -> Callable[[], Any]:
        """Get anemoi-datasets task"""
        task_func = creator_factory(name.replace("-", "_"), **opt)

        @functools.wraps(task_func)
        def wrapped_func(*prior):
            return task_func.run()

        if "parts" in opt:
            wrapped_func.__name__ = f"{name}:{opt['parts']}"
        else:
            wrapped_func.__name__ = f"{name}"
        return wrapped_func

    def apply_sequential_task(prior: fluent.Action, task_name: str, opt: dict[str, Any] | None = None) -> fluent.Action:
        """Apply a task on each node in the graph"""
        if opt is None:
            opt = options.copy()
        return prior.map(fluent.Payload(get_task(task_name, opt)))

    def apply_parallel_task(node: fluent.Action, task_name: str, dim: str) -> fluent.Action:
        """Apply a task in parallel creating a new dimension"""
        parallel_node = node.transform(
            apply_sequential_task,
            [(task_name, get_options(n)) for n in range(total)],
            dim=(dim, list(range(total))),
        )
        if dim not in parallel_node.nodes.dims:
            parallel_node.nodes = parallel_node.nodes.expand_dims(dim)
        return parallel_node

    def apply_reduction_task(node: fluent.Action, task_name: str, dim: str) -> fluent.Action:
        """Apply a task which reduces the dimension"""
        return node.reduce(fluent.Payload(get_task(task_name, options)), dim=dim)

    loaded = apply_parallel_task(init, "load", dim="parts")
    finalised = apply_reduction_task(loaded, "finalise", dim="parts")

    init_added = apply_sequential_task(finalised, "init-additions")
    load_added = apply_parallel_task(init_added, "load-additions", "parts")
    finalise_additions = apply_reduction_task(load_added, "finalise-additions", dim="parts")

    patch = apply_sequential_task(finalise_additions, "patch")
    cleanup = apply_sequential_task(patch, "cleanup")
    verify = apply_sequential_task(cleanup, "verify")

    return verify


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


__all__ = [
    "from_config",
    "from_input",
    "from_initial_conditions",
    "Action",
]
