# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Fluent API for anemoi-inference.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from earthkit.data.utils.dates import to_datetime, to_timedelta

from earthkit.workflows import fluent

from .types import ENSEMBLE_DIMENSION_NAME
from .utils import crack_environment, expansion_qube, faked_ensemble_transform, parse_ensemble_members

# from earthkit.workflows.plugins.anemoi.inference import _get_initial_conditions_source
# from earthkit.workflows.plugins.anemoi.inference import run_model
# from earthkit.workflows.plugins.anemoi.runner import CascadeRunner


if TYPE_CHECKING:
    # anemoi-inference imports
    from anemoi.inference.checkpoint import Checkpoint
    from anemoi.inference.config.run import RunConfiguration
    from anemoi.inference.metadata import Metadata
    from anemoi.inference.types import State

    from .types import DATE, ENSEMBLE_MEMBER_SPECIFICATION, ENVIRONMENT, LEAD_TIME, VALID_CKPT

LOG = logging.getLogger(__name__)


def _get_initial_conditions_source(
    config: RunConfiguration | dict | fluent.Action,
    date: DATE,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION | None = None,
    *,
    initial_condition_perturbation: bool = False,
    payload_metadata: dict[str, Any] | None = None,
) -> fluent.Action:
    """
    Get the initial conditions for the model

    Parameters
    ----------
    config : RunConfiguration | fluent.Action
        Configuration object, must contain checkpoint and input.
        If is a fluent action, the action must return the RunConfiguration object.
    date : str | tuple[int, int, int]
        Date to get initial conditions for
    ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION, optional
        Number of ensemble members to get, by default None
    initial_condition_perturbation : bool, optional
        Whether to get perturbed initial conditions, by default False
        If False, only one initial condition is returned, and
        the ensemble members are simulated by wrapping the action.
    payload_metadata : Optional[dict[str, Any]], optional
        Metadata to add to the payload, by default None

    Returns
    -------
    fluent.Action
        Fluent action of the initial conditions
    """
    ens_members = parse_ensemble_members(ensemble_members)
    if initial_condition_perturbation:
        if any(ens is None for ens in ens_members):
            raise ValueError("Ensemble members must be specified when using initial condition perturbation.")
        if isinstance(config, fluent.Action):
            init_conditions = config.transform(
                lambda x, *a: x.map(
                    fluent.Payload(
                        "earthkit.workflows.plugins.anemoi.inference._get_initial_conditions_from_config",
                        args=(fluent.Node.input_name(0)),
                        kwargs=dict(ens_num=a[0], date=date),
                        metadata=payload_metadata,
                    )
                ),
                params=ens_members,
                dim=(ENSEMBLE_DIMENSION_NAME, ens_members),
            )
            init_conditions._add_dimension("date", [to_datetime(date)])
            return init_conditions

        return fluent.from_source(
            [
                [
                    # fluent.Payload(_get_initial_conditions_ens, kwargs=dict(input=input, date=date, ens_mem=ens_mem))
                    fluent.Payload(
                        "earthkit.workflows.plugins.anemoi.inference._get_initial_conditions_from_config",
                        kwargs=dict(config=config, date=date, ens_mem=ens_mem),
                        metadata=payload_metadata,
                    )
                    for ens_mem in ens_members
                ],
            ],  # type: ignore
            coords={"date": [to_datetime(date)], ENSEMBLE_DIMENSION_NAME: ens_members},
        )

    if isinstance(config, fluent.Action):
        init_condition = fluent.Payload(
            "earthkit.workflows.plugins.anemoi.inference._get_initial_conditions_from_config",
            args=(fluent.Node.input_name(0),),
            kwargs=dict(date=date),
            metadata=payload_metadata,
        )
        single_init = config.map(init_condition)
        single_init._add_dimension("date", [to_datetime(date)])
    else:
        init_condition = fluent.Payload(
            "earthkit.workflows.plugins.anemoi.inference._get_initial_conditions_from_config",
            kwargs=dict(config=config, date=date),
            metadata=payload_metadata,
        )
        single_init = fluent.from_source(
            [
                init_condition,
            ],  # type: ignore
            coords={"date": [to_datetime(date)]},
        )

    # Wrap with empty payload to simulate ensemble members
    expanded_init = single_init.transform(
        faked_ensemble_transform,
        list(zip(ens_members)),
        (ENSEMBLE_DIMENSION_NAME, ens_members),  # type: ignore
    )
    if ENSEMBLE_DIMENSION_NAME not in expanded_init.nodes.coords:
        expanded_init.nodes = expanded_init.nodes.expand_dims(ENSEMBLE_DIMENSION_NAME)
    return expanded_init


def _run_model(
    metadata: Metadata,
    config: RunConfiguration | dict,
    input_state_source: fluent.Action,
    lead_time: LEAD_TIME,
    payload_metadata: dict[str, Any] | None = None,
    **kwargs,
) -> fluent.Action:
    """
    Run the model, expanding the results to the correct dimensions.

    Parameters
    ----------
    metadata : Metadata
        `anemoi.inference` metadata
    config : RunConfiguration | dict
        Configuration object
    input_state_source : fluent.Action
        Fluent action of initial conditions
    lead_time : LEAD_TIME
        Lead time to run out to. Can be a string,
        i.e. `1H`, `1D`, int, or a datetime.timedelta
    payload_metadata : Optional[dict[str, Any]], optional
        Metadata to add to the payload, by default None
    kwargs : dict
        Additional arguments to pass to the runner

    Returns
    -------
    fluent.Action
        Cascade action of the model results
    """
    lead_time = to_timedelta(lead_time)

    model_payload = fluent.Payload(
        "earthkit.workflows.plugins.anemoi.inference.run_as_earthkit_from_config",
        args=(fluent.Node.input_name(0),),
        kwargs=dict(config=config, lead_time=lead_time, **kwargs),
        metadata=payload_metadata,
    )

    qube = expansion_qube(metadata, lead_time)
    model_results = input_state_source.map(model_payload, yields=("step", list(qube.axes()["step"])))
    return model_results.expand_as_qube(qube.remove_by_key("step"))


def from_config(
    config: os.PathLike | dict[str, Any] | RunConfiguration,
    overrides: dict[str, Any] | None = None,
    *,
    date: DATE | None = None,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION | None = None,
    environment: ENVIRONMENT | None = None,
    **kwargs: Any,
) -> fluent.Action:
    """
    Run an anemoi-inference model from a configuration file

    Parameters
    ----------
    config : os.PathLike | dict[str, Any] | RunConfiguration
        Path to the configuration file, or dictionary of configuration
    overrides : Optional[dict[str, Any]], optional
        Override for the config, by default None
    date : Optional[DATE], optional
        Specific override for date, by default None
    ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION , optional
        Number of ensemble members to run, None will run a single instance, by default None
    environment : ENVIRONMENT, optional
        Environment to run the model in, by default None
        If None, will use the current environment
        Should be set to strings, as if used in pip install,
        e.g. `["anemoi-models==0.3.1"]`
        Can be dict[str, list[str]] with keys `inference` and `initial_conditions`
        to set the environment for each part of the run.
    kwargs : dict
        Additional arguments to pass to the configuration

    Returns
    -------
    fluent.Action
        earthkit.workflows action of the model results

    Raises
    -------
    ImportError
        Requires `anemoi-inference` installed in the creation environment
        due to validation of the config.

    Examples
    --------
    >>> from earthkit.workflows.plugins.anemoi.fluent import from_config
    >>> from_config("config.yaml", date = "2021-01-01T00:00:00")
    """
    try:
        from anemoi.inference.config.run import RunConfiguration
    except ImportError as e:
        raise ImportError("Using `from_config` requires `anemoi-inference` to be installed.") from e

    environment = crack_environment(environment, ["inference", "initial_conditions"])
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
        raise TypeError(
            f"Invalid type for config: {type(config)}. " "Must be os.PathLike, dict[str, Any], or RunConfiguration."
        )

    input_state_source = _get_initial_conditions_source(
        config=configuration,
        date=date or configuration.date,  # type: ignore
        ensemble_members=ensemble_members,
        payload_metadata={"environment": environment["initial_conditions"]},
    )

    return _run_model(
        metadata,  # TODO Figure out how to get the metadata here
        configuration,
        input_state_source,
        configuration.lead_time,
        payload_metadata={"environment": environment["inference"]},
    )


def from_input(
    ckpt: VALID_CKPT,
    input: str | dict[str, Any],
    date: DATE,
    lead_time: LEAD_TIME,
    *,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION | None = None,
    environment: ENVIRONMENT | None = None,
    **kwargs: Any,
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
    date : DATE
        Date to get initial conditions for
    lead_time : LEAD_TIME
        Lead time to run out to. Can be a string,
        i.e. `1H`, `1D`, int, or a datetime.timedelta
    ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION, optional
        Number of ensemble members to run, None will run a single instance, by default None
    environment : ENVIRONMENT, optional
        Environment to run the model in, by default None
        If None, will use the current environment
        Should be set to strings, as if used in pip install,
        e.g. `["anemoi-models==0.3.1"]`
        Can be dict[str, list[str]] with keys `inference` and `initial_conditions`
        to set the environment for each part of the run.
    kwargs : dict
        Additional arguments to pass to the configuration

    Returns
    -------
    fluent.Action
        earthkit.workflows action of the model results

    Examples
    -------
    >>> from earthkit.workflows.plugins.anemoi.fluent import from_input
    >>> from_input("anemoi_model.ckpt", "mars", date = "2021-01-01T00:00:00", lead_time = "10D")
    """
    config = dict(checkpoint=ckpt, input=input, **kwargs)
    environment = crack_environment(environment, ["inference", "initial_conditions"])

    input_state_source = _get_initial_conditions_source(
        config=config,
        date=date,
        ensemble_members=ensemble_members,
        payload_metadata={"environment": environment["initial_conditions"]},
    )

    return _run_model(
        metadata,  # TODO Figure out how to get the metadata here
        config,
        input_state_source,
        lead_time,
        payload_metadata={"environment": environment["inference"]},
    )


def from_initial_conditions(
    ckpt: VALID_CKPT,
    initial_conditions: State | fluent.Action | fluent.Payload | Callable,
    lead_time: LEAD_TIME,
    configuration_kwargs: dict[str, Any] | None = None,
    *,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION | None = None,
    environment: ENVIRONMENT | None = None,
    **kwargs: Any,
) -> fluent.Action:
    """
    Run an anemoi inference model from initial conditions

    Parameters
    ----------
    ckpt : VALID_CKPT
        Checkpoint to load
    initial_conditions : State | fluent.Action | fluent.Payload | Callable
        Initial conditions for the model
        Can be other fluent actions, payloads, or a callable, or a State.
        If a fluent action and multiple ensemble member initial conditions
        are included, the dimension must be named `ensemble_member`.
    lead_time : LEAD_TIME
        Lead time to run out to. Can be a string,
        i.e. `1H`, `1D`, int, or a datetime.timedelta
    configuration_kwargs: dict[str, Any]:
        kwargs for `anemoi.inference` configuration
    ensemble_members : Optional[ENSEMBLE_MEMBER_SPECIFICATION], optional
        Number of ensemble members to run,
        If initial_conditions is a fluent action, with
        multiple ensemble members, this argument can be set to None,
        and the number of ensemble members will be inferred from the action.
        by default None.
    environment : Optional[list[str]], optional
        Environment to run the model in, by default None
        If None, will use the current environment
        Should be set to strings, as if used in pip install,
        e.g. `["anemoi-models==0.3.1"]`
    kwargs : dict
        Additional arguments to pass to the configuration

    Returns
    -------
    fluent.Action
        earthkit.workflows action of the model results

    Examples
    --------
    >>> from earthkit.workflows.plugins.anemoi.fluent import from_initial_conditions
    >>> from_initial_conditions("anemoi_model.ckpt", init_conditions, lead_time = "10D")
    """

    config = dict(checkpoint=ckpt, **(configuration_kwargs or {}))
    environment_dict = crack_environment(environment, ["inference"])

    if isinstance(initial_conditions, fluent.Action):
        initial_conditions = initial_conditions
    elif isinstance(initial_conditions, (Callable, fluent.Payload)):
        initial_conditions = fluent.from_source([initial_conditions])  # type: ignore
    else:
        initial_conditions = fluent.from_source([fluent.Payload(lambda: initial_conditions)], dims=["date"])  # type: ignore

    if ENSEMBLE_DIMENSION_NAME in initial_conditions.nodes.dims:
        if ensemble_members is None:
            ensemble_members = len(initial_conditions.nodes.coords[ENSEMBLE_DIMENSION_NAME])

        parsed_ensemble_members = parse_ensemble_members(ensemble_members)

        if not len(initial_conditions.nodes.coords[ENSEMBLE_DIMENSION_NAME]) == len(parsed_ensemble_members):
            raise ValueError("Number of ensemble members in initial conditions must match `ensemble_members` argument")
        ens_initial_conditions = initial_conditions

    else:
        ens_initial_conditions = initial_conditions.transform(
            faked_ensemble_transform,
            list(zip(parse_ensemble_members(ensemble_members))),  # type: ignore
            (ENSEMBLE_DIMENSION_NAME, parse_ensemble_members(ensemble_members)),  # type: ignore
        )
    return _run_model(
        metadata,  # TODO Figure out how to get the metadata here
        config,
        ens_initial_conditions,
        lead_time,
        payload_metadata={"environment": environment_dict["inference"]},
    )


def create_dataset(
    config: dict[str, Any] | os.PathLike | str,
    path: os.PathLike | str,
    *,
    number_of_tasks: int | None = None,
    overwrite: bool = False,
    test: bool = False,
    environment: list[str] | None = None,
) -> fluent.Action:
    """
    Create an anemoi-dataset from a configuration.

    Parameters
    ----------
    config : dict[str, Any] | os.PathLike | sre
        Configuration to use
    path : os.PathLike | str
        Path to save the dataset to
    number_of_tasks : Optional[int], optional
        Number of tasks to run in parallel, by default None
        If None, will use a heurisitic based on date groups
    overwrite : bool, optional
        Whether to overwrite the dataset if it exists, by default False
    test : bool, optional
        Build a small dataset, using only the first dates. And, when possible, using low resolution and less ensemble members,
        by default False
    environment : Optional[list[str]], optional
        Environment to run the model in, by default None
        If None, will use the current environment
        Should be set to strings, as if used in pip install,
        e.g. `["anemoi-datasets==0.3.1"]`

    Returns
    -------
    fluent.Action
        earthkit.workflows action to create the dataset

    Raises
    -------
    ImportError
        Requires `anemoi-datasets` installed in the creation environment
        due to validation of the config.

    Examples
    --------
    >>> from earthkit.workflows.plugins.anemoi.fluent import create_dataset
    >>> create_dataset("dataset_recipe.yaml", "output_dir/dataset.zarr")
    """
    import yaml

    try:
        from anemoi.datasets.create import creator_factory
    except ImportError as e:
        raise ImportError("Using `create_dataset` requires `anemoi-datasets` to be installed.") from e

    config_path, config_dict = None, None

    if isinstance(config, dict):
        import tempfile

        temp_config_file = tempfile.NamedTemporaryFile(suffix=".yaml")
        yaml.dump(config, open(temp_config_file.name, "w"))
        config_path = temp_config_file.name
        config_dict = config.copy()
    else:
        config_path = os.path.realpath(config)
        config_dict = yaml.safe_load(open(config_path))

    if number_of_tasks is None:
        from anemoi.datasets.create.config import loader_config
        from anemoi.datasets.dates.groups import Groups

        groups = Groups(**loader_config(config_dict.copy()).dates)
        number_of_tasks = len(groups) * 25

    path = os.path.abspath(path)
    options = {"config": config_path, "path": path, "overwrite": overwrite, "test": test}
    payload_metadata = {"environment": environment or []}

    def get_parallel_options(part: int):
        opt = options.copy()
        opt["parts"] = f"{part+1}/{number_of_tasks}"
        return opt

    def get_task(name: str, opt: dict[str, Any]) -> Callable[..., Any]:
        """Get anemoi-datasets task"""

        def wrapped_func(*prior):
            task_func = creator_factory(name.replace("-", "_"), **opt)
            return task_func.run()

        if "parts" in opt:
            wrapped_func.__name__ = f"{name}:{opt['parts']}"
        else:
            wrapped_func.__name__ = f"{name}"
        return wrapped_func

    def get_payload(task: Callable[..., Any]) -> fluent.Payload:
        """Get fluent payload"""
        return fluent.Payload(task, metadata=payload_metadata)

    def apply_sequential_task(prior: fluent.Action, task_name: str, opt: dict[str, Any] | None = None) -> fluent.Action:
        """Apply a task on each node in the graph"""
        if opt is None:
            opt = options.copy()
        return prior.map(get_payload(get_task(task_name, opt)))

    def apply_parallel_task(node: fluent.Action, task_name: str, dim: str) -> fluent.Action:
        """Apply a task in parallel creating a new dimension"""
        parallel_node = node.transform(
            apply_sequential_task,
            [(task_name, get_parallel_options(n)) for n in range(number_of_tasks)],
            dim=(dim, list(range(number_of_tasks))),
        )
        if dim not in parallel_node.nodes.dims:
            parallel_node.nodes = parallel_node.nodes.expand_dims(dim)
        return parallel_node

    def apply_reduction_task(node: fluent.Action, task_name: str, dim: str) -> fluent.Action:
        """Apply a task which reduces the dimension"""
        return node.reduce(get_payload(get_task(task_name, options)), dim=dim)

    init = fluent.from_source([get_payload(get_task("init", options.copy()))], dims=["source"])  # type: ignore
    loaded = apply_parallel_task(init, "load", dim="parts")
    finalised = apply_reduction_task(loaded, "finalise", dim="parts")

    init_added = apply_sequential_task(finalised, "init-additions")
    load_added = apply_parallel_task(init_added, "load-additions", "parts")
    finalise_additions = apply_reduction_task(load_added, "finalise-additions", dim="parts")

    patch = apply_sequential_task(finalise_additions, "patch")
    cleanup = apply_sequential_task(patch, "cleanup")
    verify = apply_sequential_task(cleanup, "verify")

    def get_path(_):
        return path

    return verify.map(get_path)


def from_dataset(
    ckpt: VALID_CKPT,
    dataset_config: dict[str, Any] | os.PathLike,
    date: DATE,
    lead_time: LEAD_TIME,
    *,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION | None = None,
    input_template: dict[str, Any] | None = None,
    number_of_dataset_tasks: int | None = None,
    environment: ENVIRONMENT | None = None,
    **kwargs: Any,
) -> fluent.Action:
    """
    Run an anemoi inference model after creating a dataset from a recipe.

    NOTE: This will create a dataset in the current working directory
    and will not delete it after the run. #TODO Fix.

    Parameters
    ----------
    ckpt : VALID_CKPT
        Checkpoint to load
    dataset_config : dict[str, Any] | os.PathLike
        Dataset Configuration to use
    date : DATE
        Date to get initial conditions for
    lead_time : LEAD_TIME
        Lead time to run out to. Can be a string,
        i.e. `1H`, `1D`, int, or a datetime.timedelta
    ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION, optional
        Number of ensemble members to run, None will run a single instance, by default None
    input_template : Optional[dict[str, Any]], optional
        Template of input to use for inference, by default None
        Use "%DATASET_PATH%" to mark where the dataset path should be inserted
        Default is {"dataset": "%DATASET_PATH%"}
    number_of_dataset_tasks : Optional[int], optional
        Number of tasks to run in parallel, by default None
        If None, will use a heurisitic based on date groups
    environment : ENVIRONMENT, optional
        Environment to run the model in, by default None
        If None, will use the current environment
        Should be set to strings, as if used in pip install,
        e.g. `["anemoi-models==0.3.1"]`
        Can be dict[str, list[str]] with keys `inference`, `initial_conditions`, and `dataset`
        to set the environment for each part of the run.
    kwargs : dict
        Additional arguments to pass to the runner

    Returns
    -------
    fluent.Action
        earthkit.workflows actions of the dataset, and then model.

    Examples
    -------
    >>> from earthkit.workflows.plugins.anemoi.fluent import from_dataset
    >>> from_dataset("anemoi_model.ckpt", "dataset_recipe.yaml", date = "2021-01-01T00:00:00", lead_time = "10D")
    """
    import tempfile

    import yaml

    if not isinstance(dataset_config, os.PathLike):
        dataset_config = yaml.safe_load(open(dataset_config))

    checkpoint = Checkpoint(ckpt)

    dates = list(map(lambda x: to_datetime(date) + x, checkpoint.lagged))
    end_date = max(dates)
    start_date = min(dates)
    frequency = checkpoint.timestep

    dataset_config["dates"] = {
        "start": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "end": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "frequency": str(frequency),
    }

    temp_config_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
    yaml.dump(dataset_config, open(temp_config_file.name, "w"))

    runner_config = dict(checkpoint=ckpt, input="dummy", **kwargs)

    environment = crack_environment(environment, ["inference", "dataset", "initial_conditions"])

    def construct_configuration(dataset_location: str):
        """Create configuration from dataset location"""

        def insert_dataset_path(template: dict[str, Any] | list[str] | str) -> dict | list | str:
            """Insert the dataset path into the template"""
            if isinstance(template, dict):
                return {k: insert_dataset_path(v) for k, v in template.items()}
            elif isinstance(template, list):
                return [insert_dataset_path(v) for v in template]
            elif isinstance(template, str):
                if "%DATASET_PATH%" in template:
                    return template.replace("%DATASET_PATH%", dataset_location)
                else:
                    return template
            return template

        dataset_input = insert_dataset_path(input_template or {"dataset": "%DATASET_PATH%"})

        config = dict(
            checkpoint=ckpt,
            input=dataset_input,
            **kwargs,
        )
        return config

    dataset_action = create_dataset(
        temp_config_file.name,
        f"dataset-{date}-for_inference.zarr",
        overwrite=True,
        number_of_tasks=number_of_dataset_tasks,
        environment=environment["dataset"],
    )
    init_conditions_config = dataset_action.map(
        fluent.Payload(construct_configuration, args=(fluent.Node.input_name(0),))
    )

    input_state_source = _get_initial_conditions_source(
        config=init_conditions_config,
        date=date,
        ensemble_members=ensemble_members,
        payload_metadata={"environment": environment["initial_conditions"]},
    )
    return _run_model(
        metadata,  # TODO Figure out how to get the metadata here
        runner_config,
        input_state_source,
        lead_time,
        payload_metadata={"environment": environment["inference"]},
    )


class Action(fluent.Action):
    """Anemoi Fluent Action"""

    def infer(
        self,
        ckpt: VALID_CKPT,
        lead_time: LEAD_TIME,
        configuration_kwargs: dict[str, Any] | None = None,
        environment: ENVIRONMENT | None = None,
        **kwargs,
    ) -> fluent.Action:
        """
        Map a model prediction to all nodes within the graph, using them as initial conditions.

        Parameters
        ----------
        ckpt : VALID_CKPT
            Checkpoint to load
        lead_time : LEAD_TIME
            Lead time to run out to. Can be a string,
            i.e. `1H`, `1D`, int, or a datetime.timedelta
        configuration_kwargs: dict[str, Any]:
            kwargs for anemoi.inference configuration
        environment : Optional[list[str]], optional
            Environment to run the model in, by default None
            If None, will use the current environment
            Should be set to strings, as if used in pip install,
            e.g. `["anemoi-models==0.3.1"]`
        kwargs : dict
            Additional arguments to pass to the runner


        Returns
        -------
        fluent.Action
            Cascade action of the model results
        """
        return from_initial_conditions(
            ckpt, self, lead_time, configuration_kwargs=configuration_kwargs, environment=environment, **kwargs
        )


fluent.Action.register("anemoi", Action)


__all__ = [
    "from_config",
    "from_input",
    "from_initial_conditions",
    "create_dataset",
    "from_dataset",
    "Action",
]
