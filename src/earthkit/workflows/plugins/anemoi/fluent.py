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
from functools import reduce
from operator import or_
from typing import TYPE_CHECKING, Any, TypeVar

from anemoi.utils.dates import as_timedelta
from earthkit.data.utils.dates import to_datetime
from earthkit.workflows.metadata import Artifacts, Requirements
from qubed import Qube

from earthkit.workflows import fluent

from .types import ENSEMBLE_DIMENSION_NAME
from .utils import crack_environment, expansion_qube_from_metadata, parse_ensemble_members

if TYPE_CHECKING:
    # anemoi-inference imports
    from anemoi.inference.config.run import RunConfiguration
    from anemoi.inference.metadata import Metadata
    from anemoi.inference.types import State

    from .types import DATE, ENSEMBLE_MEMBER_SPECIFICATION, ENVIRONMENT, LEAD_TIME, VALID_CKPT

LOG = logging.getLogger(__name__)
T = TypeVar("T")
DEFAULT_DATASET_NAME = "data"


def _get_metadata(ckpt: VALID_CKPT, *, metadata: dict[str, Metadata] | None = None) -> dict[str, Metadata]:
    if metadata is not None:
        return metadata

    from anemoi.inference.checkpoint import Checkpoint

    checkpoint = Checkpoint(ckpt)  # type: ignore
    return checkpoint.multi_dataset_metadata  # type: ignore[reportReturnType]


def _get_initial_conditions_source(
    config: dict | fluent.Action | RunConfiguration,
    date: DATE,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION | None = None,
    *,
    requirements: Requirements | None = None,
    artifacts: Artifacts | None = None,
) -> fluent.Action:
    """
    Get the initial conditions for the model

    Parameters
    ----------
    config : dict | fluent.Action | RunConfiguration
        Configuration object, must contain checkpoint and input.
        If is a fluent action, the action must return the RunConfiguration object.
    date : str | tuple[int, int, int]
        Date to get initial conditions for
    ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION, optional
        Number of ensemble members to get, by default None
    requirements : Requirements | None, optional
        Requirements for running the model, by default None
    artifacts : Artifacts | None, optional
        Artifacts required for the model, by default None

    Returns
    -------
    fluent.Action
        Fluent action of the initial conditions
    """
    # Convert RunConfiguration to dict for payload execution (but keep Action as-is)
    if hasattr(config, "model_dump") and callable(getattr(config, "model_dump")):
        config_dict: dict = config.model_dump()  # type: ignore
    else:
        config_dict = config

    if ensemble_members is None:
        ensemble_members = [0]

    ens_members = parse_ensemble_members(ensemble_members)

    if any(ens is None for ens in ens_members):
        raise ValueError("Ensemble members must be specified when using initial condition perturbation.")
    with fluent.NodeMetadataContext(requirements=requirements, artifacts=artifacts):
        if isinstance(config_dict, fluent.Action):
            init_conditions = config_dict.transform(
                lambda x, *a: x.map(
                    fluent.create_task_instance(
                        "earthkit.workflows.plugins.anemoi.inference._get_initial_conditions",
                        static_input_ps=(fluent.Node.Index(0)),
                        static_input_kw=dict(number=a[0], date=date),
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
                    fluent.create_task_instance(
                        "earthkit.workflows.plugins.anemoi.inference._get_initial_conditions",
                        static_input_ps=(),
                        static_input_kw=dict(config=config_dict, date=date, number=ens_mem),
                    )
                    for ens_mem in ens_members
                ],
            ],  # type: ignore
            coords={"date": [to_datetime(date)], ENSEMBLE_DIMENSION_NAME: ens_members},
        )


def _run_model(
    expansion_qube: dict[str, Qube],
    config: RunConfiguration | dict,
    input_state_source: fluent.Action,
    lead_time: LEAD_TIME,
    requirements: Requirements | None = None,
    artifacts: Artifacts | None = None,
    **kwargs,
) -> fluent.Action:
    """
    Run the model, expanding the results to the correct dimensions.

    Parameters
    ----------
    expansion_qube : dict[str, Qube]
        Expansion qube for the model, can be got from the metadata using `utils.expansion_qube_from_metadata`
    config : RunConfiguration | dict
        Configuration object
    input_state_source : fluent.Action
        Fluent action of initial conditions
    lead_time : LEAD_TIME
        Lead time to run out to. Can be a string,
        i.e. `1H`, `1D`, int, or a datetime.timedelta
    requirements : Requirements | None, optional
        Requirements for running the model, by default None
    artifacts : Artifacts | None, optional
        Artifacts required for the model, by default None
    kwargs : dict
        Additional arguments to pass to the runner

    Returns
    -------
    fluent.Action
        Cascade action of the model results,
        Includes a dataset branch structure, based on the keys of the expansion qube, and the results are expanded along the dimensions of the corresponding qube.
    """
    # Convert RunConfiguration to dict for payload execution
    config_dict = (
        config.model_dump()  # type: ignore
        if hasattr(config, "model_dump") and callable(getattr(config, "model_dump"))
        else config
    )

    model_payload = fluent.create_task_instance(
        "earthkit.workflows.plugins.anemoi.inference.run_as_earthkit",
        static_input_ps=(fluent.Node.Index(0),),
        static_input_kw=dict(config=config_dict, lead_time=as_timedelta(lead_time), **kwargs),
        requirements=Requirements(needs_gpu=True),
    )

    # Qube axes are unordered sets, but the runner yields states in chronological order,
    # so the step coordinates must be sorted to label each yielded state correctly.
    step_dimension = sorted(next(iter(expansion_qube.values())).axes()["step"], key=as_timedelta)
    expansion_qubes_no_step = {k: v.remove_by_key("step") for k, v in expansion_qube.items()}

    with fluent.NodeMetadataContext(requirements=requirements, artifacts=artifacts):
        model_results = input_state_source.map(model_payload, yields=("step", step_dimension))

    dataset_qube = reduce(or_, (f"dataset={ds}" / q for ds, q in expansion_qubes_no_step.items()))
    return model_results.expand_as_qube(dataset_qube)


def _default_dictionarify(obj: Any | dict[str, T | Any], expected_type: type[T]) -> dict[str, T]:
    if isinstance(obj, expected_type):
        return {DEFAULT_DATASET_NAME: obj}
    elif isinstance(obj, dict):
        if not any(isinstance(v, expected_type) for v in obj.values()):
            return {DEFAULT_DATASET_NAME: obj}
        elif all(isinstance(v, expected_type) for v in obj.values()):
            return obj
        else:
            raise ValueError(
                f"Invalid dictionary, values must all be of type {expected_type} or none of them must be of type {expected_type}. Got: {obj}"
            )
    raise TypeError(f"Invalid type for obj: {type(obj)}. Expected {expected_type} or dict[str, {expected_type}].")


class Inference:
    """Build fluent workflow actions for one anemoi inference setup."""

    def __init__(
        self,
        ckpt: VALID_CKPT,
        lead_time: LEAD_TIME,
        *,
        environment: ENVIRONMENT | None = None,
        metadata: dict[str, Metadata] | None = None,
        expansion_qube: Qube | dict[str, Qube] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        ckpt : VALID_CKPT
            Checkpoint to load
        lead_time : LEAD_TIME
            Lead time to run out to. Can be a string,
            i.e. `1H`, `1D`, int, or a datetime.timedelta
        environment : ENVIRONMENT, optional
            Environment to run the model in, by default None
        metadata : dict[str, Metadata] | None, optional
            `anemoi.inference` multi metadata, if not given will be got from the checkpoint on disk, by default None
        expansion_qube : Qube | None, optional
            Qube to expand the model by, if not given will be got from the metadata using `utils.expansion_qube_from_metadata`, by default None
        kwargs : dict
            Additional arguments to pass to the runner configuration
        """
        self.ckpt = ckpt
        self.lead_time = lead_time
        self.environment: ENVIRONMENT = environment if environment is not None else []

        self._given_single_qube = isinstance(expansion_qube, Qube)

        self.expansion_qube = (
            _default_dictionarify(expansion_qube, Qube)
            if expansion_qube is not None
            else expansion_qube_from_metadata(_get_metadata(ckpt, metadata=metadata), as_timedelta(lead_time))
        )
        self.kwargs = kwargs

    def _config(self, **kwargs: Any) -> dict[str, Any]:
        if set(kwargs.keys()) & {"checkpoint", *self.kwargs.keys()}:
            raise ValueError(
                f"Cannot override checkpoint or keys set in the constructor: {set(kwargs.keys()) & {'checkpoint', *self.kwargs.keys()}}"
            )
        return {"checkpoint": self.ckpt, **self.kwargs, **kwargs}

    def _run_model(
        self,
        config: dict,
        input_state_source: fluent.Action,
        *,
        requirements: Requirements | None = None,
        artifacts: Artifacts | None = None,
        **kwargs: Any,
    ) -> fluent.Action:
        model_results = _run_model(
            self.expansion_qube,
            config,
            input_state_source,
            self.lead_time,
            requirements=requirements,
            artifacts=artifacts,
            **kwargs,
        )
        if self._given_single_qube:
            model_results = model_results.select({"dataset": next(iter(self.expansion_qube.keys()))})
        return model_results

    def from_input(
        self,
        input: str | dict[str, Any],
        date: DATE,
        *,
        ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION | None = None,
        requirements: Requirements | None = None,
        artifacts: Artifacts | None = None,
        **kwargs: Any,
    ) -> fluent.Action:
        """
        Run an anemoi inference model from a given input source.

        Parameters
        ----------
        input : str | dict[str, Any]
            `anemoi.inference` input source.
            Can be `mars`, `grib`, etc or a dictionary of input configuration,
        date : DATE
            Date to get initial conditions for
        ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION, optional
            Number of ensemble members to run, None will run a single instance, by default None
        requirements : Requirements | None, optional
            Requirements for running the model, by default None
        artifacts : Artifacts | None, optional
            Artifacts required for the model, by default None
        kwargs : dict
            Additional arguments to pass to the runner configuration

        Returns
        -------
        fluent.Action
            earthkit.workflows action of the model results

        Examples
        -------
        >>> from earthkit.workflows.plugins.anemoi.fluent import Inference
        >>> inference = Inference("anemoi_model.ckpt", lead_time = "10D")
        >>> inference.from_input("mars", date = "2021-01-01T00:00:00")
        """
        config = self._config(input=input, **kwargs)
        environment_dict = crack_environment(
            self.environment,
            ["inference", "initial_conditions"],
        )

        input_state_source = _get_initial_conditions_source(
            config=config,
            date=date,
            ensemble_members=ensemble_members,
            requirements=Requirements(environment=environment_dict["initial_conditions"]),
        )

        with fluent.NodeMetadataContext(
            requirements=requirements,
            artifacts=artifacts,
        ):
            return self._run_model(
                config,
                input_state_source,
                requirements=Requirements(environment=environment_dict["inference"]),
            )

    def from_initial_conditions(
        self,
        initial_conditions: dict[str, State] | None | fluent.Action | fluent.Payload | Callable,
        requirements: Requirements | None = None,
        artifacts: Artifacts | None = None,
        **kwargs: Any,
    ) -> fluent.Action:
        """
        Run an anemoi inference model from initial conditions.

        Parameters
        ----------
        initial_conditions : dict[str, State] | None | fluent.Action | fluent.Payload | Callable
            Initial conditions for the model
            Can be other fluent actions, payloads, a callable, a State, or None.
            None creates a source node that yields None.
            If a fluent action and multiple ensemble member initial conditions
            are included, the dimension must be named `ensemble_member`.
        requirements : Requirements | None, optional
            Requirements for running the model, by default None
        artifacts : Artifacts | None, optional
            Artifacts required for the model, by default None
        kwargs : dict
            Additional arguments to pass to the runner configuration

        Returns
        -------
        fluent.Action
            earthkit.workflows action of the model results

        Examples
        --------
        >>> from earthkit.workflows.plugins.anemoi.fluent import Inference
        >>> inference = Inference("anemoi_model.ckpt", lead_time = "10D")
        >>> inference.from_initial_conditions(init_conditions)
        """
        config = self._config(**kwargs)
        environment_dict = crack_environment(self.environment, ["inference"])

        if isinstance(initial_conditions, fluent.Action):
            initial_conditions_source = initial_conditions
        elif isinstance(initial_conditions, (Callable, fluent.Payload)):
            initial_conditions_source = fluent.from_source([initial_conditions])  # type: ignore
        else:
            initial_conditions_source = fluent.from_source(
                [fluent.create_task_instance(lambda: initial_conditions)],
                dims=["date"],
            )  # type: ignore

        with fluent.NodeMetadataContext(
            requirements=requirements,
            artifacts=artifacts,
        ):
            return self._run_model(
                config,
                initial_conditions_source,
                requirements=Requirements(environment=environment_dict["inference"]),
            )


def from_config(
    config: os.PathLike | dict[str, Any] | RunConfiguration,
    overrides: dict[str, Any] | None = None,
    *,
    date: DATE | None = None,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION | None = None,
    environment: ENVIRONMENT | None = None,
    requirements: Requirements | None = None,
    artifacts: Artifacts | None = None,
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
    requirements : Requirements | None, optional
        Requirements for running the model, by default None
    artifacts : Artifacts | None, optional
        Artifacts required for the model, by default None
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
            f"Invalid type for config: {type(config)}. Must be os.PathLike, dict[str, Any], or RunConfiguration."
        )

    input_state_source = _get_initial_conditions_source(
        config=configuration,
        date=date or configuration.date,  # type: ignore
        ensemble_members=ensemble_members,
        requirements=Requirements(environment=environment["initial_conditions"]),
    )

    with fluent.NodeMetadataContext(
        requirements=requirements,
        artifacts=artifacts,
    ):
        return _run_model(
            expansion_qube_from_metadata(
                _get_metadata(configuration.checkpoint),  # type: ignore[reportArgumentType]
                as_timedelta(configuration.lead_time),
            ),  # type: ignore
            configuration,
            input_state_source,
            configuration.lead_time,
            requirements=Requirements(environment=environment["inference"]),
        )


def from_input(
    ckpt: VALID_CKPT,
    input: str | dict[str, Any],
    date: DATE,
    lead_time: LEAD_TIME,
    *,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION | None = None,
    environment: ENVIRONMENT | None = None,
    metadata: dict[str, Metadata] | None = None,
    expansion_qube: Qube | dict[str, Qube] | None = None,
    requirements: Requirements | None = None,
    artifacts: Artifacts | None = None,
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
    metadata : Optional[dict[str, Metadata]], optional
        `anemoi.inference` metadata, if not given will be got from the checkpoint on disk, by default None
    expansion_qube : Qube | dict[str, Qube] | None, optional
        Qube to expand the model by, if not given will be got from the metadata using `utils.expansion_qube_from_metadata`, by default None
    requirements : Requirements | None, optional
        Requirements for running the model, by default None
    artifacts : Artifacts | None, optional
        Artifacts required for the model, by default None
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
    return Inference(
        ckpt,
        lead_time,
        environment=environment,
        metadata=metadata,
        expansion_qube=expansion_qube,
        **kwargs,
    ).from_input(
        input,
        date,
        ensemble_members=ensemble_members,
        requirements=requirements,
        artifacts=artifacts,
    )


def from_initial_conditions(
    ckpt: VALID_CKPT,
    initial_conditions: State | None | fluent.Action | fluent.Payload | Callable,
    lead_time: LEAD_TIME,
    *,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION | None = None,
    environment: ENVIRONMENT | None = None,
    metadata: dict[str, Metadata] | None = None,
    expansion_qube: Qube | dict[str, Qube] | None = None,
    requirements: Requirements | None = None,
    artifacts: Artifacts | None = None,
    **kwargs: Any,
) -> fluent.Action:
    """
    Run an anemoi inference model from initial conditions

    Parameters
    ----------
    ckpt : VALID_CKPT
        Checkpoint to load
    initial_conditions : State | None | fluent.Action | fluent.Payload | Callable
        Initial conditions for the model
        Can be other fluent actions, payloads, a callable, a State, or None.
        None creates a source node that yields None.
        If a fluent action and multiple ensemble member initial conditions
        are included, the dimension must be named `ensemble_member`.
    lead_time : LEAD_TIME
        Lead time to run out to. Can be a string,
        i.e. `1H`, `1D`, int, or a datetime.timedelta
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
    metadata : Optional[dict[str, Metadata]], optional
        `anemoi.inference` metadata, if not given will be got from the checkpoint on disk, by default None
    expansion_qube : Qube | dict[str, Qube] | None, optional
        Qube to expand the model by, if not given will be got from the metadata using `utils.expansion_qube_from_metadata`, by default None
    requirements : Requirements | None, optional
        Requirements for running the model, by default None
    artifacts : Artifacts | None, optional
        Artifacts required for the model, by default None
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
    return Inference(
        ckpt,
        lead_time,
        environment=environment,
        metadata=metadata,
        expansion_qube=expansion_qube,
        **kwargs,
    ).from_initial_conditions(
        initial_conditions,
        ensemble_members=ensemble_members,
        requirements=requirements,
        artifacts=artifacts,
    )


def get_initial_conditions(
    ckpt: VALID_CKPT,
    input: str | dict[str, Any],
    date: DATE,
    ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION | None = None,
    *,
    environment: ENVIRONMENT | None = None,
    requirements: Requirements | None = None,
    artifacts: Artifacts | None = None,
    **kwargs: Any,
) -> fluent.Action:
    """
    Get initial conditions for an anemoi inference model.

    Retrieves the initial conditions (model state) needed to start an
    ``anemoi-inference`` run, without performing the inference itself.
    This is useful when you want to obtain the initial state separately,
    for example to inspect it, pass it to multiple models, or use it as
    input to a custom workflow.

    The checkpoint is used to determine which variables and grid the
    initial conditions should contain. The ``input`` parameter specifies
    the data source (e.g. ``"mars"``, ``"grib"``, or a dictionary of
    input configuration), matching the input sources supported by
    ``anemoi-inference``.

    Parameters
    ----------
    ckpt : VALID_CKPT
        Checkpoint to load. Can be a path to a checkpoint file, a string,
        or a dictionary (e.g. ``{"huggingface": "ecmwf/aifs-single-1.0"}``).
    input : str | dict[str, Any]
        ``anemoi-inference`` input source.
        Can be ``"mars"``, ``"grib"``, etc. or a dictionary of input
        configuration, following the same conventions as
        ``anemoi-inference``.
    date : DATE
        Date for the initial conditions. Can be a ``datetime.datetime``,
        an ISO 8601 string (e.g. ``"2021-01-01T00:00:00"``), or a
        ``(year, month, day)`` tuple.
    ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION, optional
        Number of ensemble members to retrieve, or specification of which members to retrieve.
        This can be set to None to retrieve a single member, or to a specification (e.g. an int for number of members, or a list of member identifiers) to retrieve multiple members.
        By default, None (retrieves a single member if supported).
    environment : ENVIRONMENT, optional
        Environment to run the initial conditions retrieval in, by default None.
        If None, will use the current environment.
        Should be set to strings, as if used in pip install,
        e.g. ``["anemoi-models==0.3.1"]``.
        Can be ``dict[str, list[str]]`` with the key ``"initial_conditions"``
        to set the environment for the retrieval.
    requirements : Requirements | None, optional
        Requirements for running the model, by default None
    artifacts : Artifacts | None, optional
        Artifacts required for the model, by default None
    **kwargs : dict
        Additional arguments to pass to the run configuration
        (e.g. extra input overrides).

    Returns
    -------
    fluent.Action
        An ``earthkit.workflows`` action representing the initial
        conditions. This action can be passed to
        :meth:`Inference.from_initial_conditions` or used independently.

    Examples
    --------
    Retrieve initial conditions from MARS:

    >>> from earthkit.workflows.plugins.anemoi.fluent import get_initial_conditions
    >>> ic = get_initial_conditions("anemoi_model.ckpt", "mars", date="2021-01-01T00:00:00")

    Use the initial conditions in a subsequent inference run:

    >>> from earthkit.workflows.plugins.anemoi.fluent import Inference
    >>> inference = Inference("anemoi_model.ckpt", lead_time="10D")
    >>> forecast = inference.from_initial_conditions(ic)

    Use a HuggingFace checkpoint with a dictionary input configuration:

    >>> ic = get_initial_conditions(
    ...     {"huggingface": "ecmwf/aifs-single-1.0"},
    ...     {"source": "mars", "class": "od"},
    ...     date="2022-06-15T12:00:00",
    ... )
    """
    config = {"checkpoint": ckpt, "input": input, **kwargs}

    environment_dict = crack_environment(environment, ["initial_conditions"])
    with fluent.NodeMetadataContext(
        requirements=requirements,
        artifacts=artifacts,
    ):
        return _get_initial_conditions_source(
            config=config,
            date=date,
            ensemble_members=ensemble_members,
            requirements=Requirements(environment=environment_dict["initial_conditions"]),
        )


class Action(fluent.Action):
    """Anemoi Fluent Action"""

    def infer(
        self,
        ckpt: VALID_CKPT,
        lead_time: LEAD_TIME,
        *,
        ensemble_members: ENSEMBLE_MEMBER_SPECIFICATION | None = None,
        metadata: dict[str, Metadata] | None = None,
        expansion_qube: Qube | dict[str, Qube] | None = None,
        environment: ENVIRONMENT | None = None,
        requirements: Requirements | None = None,
        artifacts: Artifacts | None = None,
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
        ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION | None, optional
            Number of ensemble members to run, If set to None, the number of ensemble members will be inferred from the action. by default None.
        metadata : dict[str, Metadata] | None, optional
            `anemoi.inference` metadata, if not given will be got from the checkpoint on disk, by default None
        expansion_qube : Qube | dict[str, Qube] | None, optional
            Qube to expand the model by, if not given will be got from the metadata using `utils.expansion_qube_from_metadata`, by default None
        environment : Optional[list[str]], optional
            Environment to run the model in, by default None
            If None, will use the current environment
            Should be set to strings, as if used in pip install,
            e.g. `["anemoi-models==0.3.1"]`
        requirements : Requirements | None, optional
            Requirements for running the model, by default None
        artifacts : Artifacts | None, optional
            Artifacts required for the model, by default None
        kwargs : dict
            Additional arguments to pass to the configuration


        Returns
        -------
        fluent.Action
            Cascade action of the model results
        """
        return from_initial_conditions(
            ckpt,
            self,
            lead_time,
            ensemble_members=ensemble_members,
            environment=environment,
            metadata=metadata,
            expansion_qube=expansion_qube,
            requirements=requirements,
            artifacts=artifacts,
            **kwargs,
        )


fluent.Action.register("anemoi", Action)


__all__ = [
    "Inference",
    "from_config",
    "from_input",
    "from_initial_conditions",
    "Action",
]
