# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import functools
import operator
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from anemoi.utils.dates import frequency_to_seconds
from earthkit.workflows.metadata import Requirements
from qubed import Qube

from earthkit.workflows import fluent

from .types import ENSEMBLE_DIMENSION_NAME, ENVIRONMENT, STAGES

if TYPE_CHECKING:
    from anemoi.inference.metadata import Metadata

    from .types import ENSEMBLE_MEMBER_SPECIFICATION, ENVIRONMENT, LEAD_TIME, REQUIREMENTS

E = TypeVar("E", bound=ENVIRONMENT)


def expansion_qube_from_metadata(metadata: "dict[str, Metadata]", lead_time: "LEAD_TIME") -> dict[str, Qube]:
    """Create a Qube object from model metadata and lead time.

    This function constructs a Qube object by analysing the model's metadata
    to identify surface, pressure level, and model level variables. It creates a
    hierarchical qube structure organising variables by their vertical coordinate
    type and expands them across time steps.

    Parameters
    ----------
    metadata : dict[str, Metadata]
        Model metadata containing variable definitions, including their vertical
        coordinate information (surface, pressure levels, model levels) and the
        model's time step. The dict keys are dataset names and the values must be
        `anemoi.inference.metadata.Metadata` instances.
        The forecast lead time as an integer or string (e.g., "7D" for 7 days).
        This determines the number of time steps in the expansion.
        If an integer is provided, it is interpreted as hours.
    lead_time : LEAD_TIME
        The forecast lead time as an integer or string (e.g., "7D" for 7 days). This determines the number of time steps in the expansion.
        If an integer is provided, it is interpreted as hours.

    Returns
    -------
    dict[str, Qube]
        A dictionary of qube objects with a hierarchical structure containing up to three
        branches: surface, pressure, and model level variables. Each branch
        contains the appropriate parameters and coordinates.

    Notes
    -----
    The function creates a qube with the following structure:

    - Surface variables: expanded over (step, param)
    - Pressure level variables: expanded over (step, param, level)
    - Model level variables: expanded over (step, param, level)

    Only variables marked as "diagnostic" or "prognostic" that don't have MARS
    requests are included. The time steps are calculated from the model's time
    step size up to the specified lead time.

    Examples
    --------
    Create expansion coordinates for a 5-day forecast:

    >>> from anemoi.inference.checkpoint import Checkpoint
    >>> ckpt = Checkpoint("path/to/checkpoint.ckpt")
    >>> metadata = ckpt.metadata
    >>> qube = expansion_qube(metadata, "5D")
    >>> qube.axes()
    {'step': {6, 12, 18, ..., 120}, 'param': {...}, 'level': {...}}

    Use with an action to expand across all dimensions:

    >>> expanded_action = action.expand_as_qube(qube)

    See Also
    --------
    Qube : The Qube class for manual qube construction
    """

    qubes = {}
    for dataset, md in metadata.items():
        variables = md.select_variables(include=["diagnostic", "prognostic"], has_mars_requests=False)
        variables_metadata = md.typed_variables
        model_step = md.timestep.seconds
        qubes[dataset] = _expansion_qube(variables, variables_metadata, model_step, lead_time)

    return qubes


def expansion_qube_from_variables(
    variables: dict[str, list[str]],
    variables_metadata: dict[str, dict[str, dict]],
    model_step: int,
    lead_time: "LEAD_TIME",
) -> dict[str, Qube]:
    """Create a Qube object from a list of variable names, their metadata, model step, and lead time.

    This function constructs a Qube object by analysing the provided list of variable names
    and their corresponding metadata to identify surface, pressure level, and model level variables.
    It creates a hierarchical qube structure organising variables by their vertical coordinate
    type and expands them across time steps.

    Parameters
    ----------
    variables : dict[str, list[str]]
        A dictionary mapping dataset names to lists of variable names to include in the qube. These should correspond to keys in the
        variables_metadata dictionary.
    variables_metadata : dict[str, dict[str, dict]]
        A dictionary mapping variable names to their metadata objects. Each metadata object should
        contain information about whether the variable is a surface level, pressure level, or model level variable,
        as well as its parameter and level information.
        Structure: {dataset_name: {variable_name: variable_metadata}}
    model_step : int
        The model's time step in seconds. This is used to calculate the time steps for expansion.
    lead_time : LEAD_TIME
        The forecast lead time as an integer or string (e.g., "7D" for 7 days). This determines the number of time steps in the expansion.
        If an integer is provided, it is interpreted as hours.

    Returns
    -------
    dict[str, Qube]
        A dictionary mapping dataset names to qube objects with a hierarchical structure containing up to three branches: surface, pressure, and model level variables. Each branch contains the appropriate parameters and coordinates.

    Notes
    -----
    The function creates a qube with the following structure:

    - Surface variables: expanded over (step, param)
    - Pressure level variables: expanded over (step, param, level)
    - Model level variables: expanded over (step, param, level)

    Only variables that are present in the provided list and have corresponding metadata are included. The time steps are calculated from the model's time step size up to the specified lead time.

    Examples
    --------
    Create expansion coordinates for specific variables:

    >>> variables = ["2t", "msl", "10u"]
    >>> variables_metadata = {
            "2t": ...,
            "msl": ...,
            "10u": ...,
        }
    >>> model_step = 3600  # 1 hour in seconds
    >>> lead_time = "5D"
    >>> qube = expansion_qube_from_variables(variables, variables_metadata, model_step, lead_time)
    >>> qube.axes()
    {'step': {6, 12, 18, ..., 120}, 'param': {130, 134, 165}, 'level': {500, 1000}}

    See Also
    --------
    Qube : The Qube class for manual qube construction
    expansion_qube_from_metadata : Create a Qube object directly from model metadata.
    """
    return {
        dataset: _expansion_qube(vars, variables_metadata[dataset], model_step, lead_time)
        for dataset, vars in variables.items()
    }


def _expansion_qube(
    variables: list[str],
    variables_metadata: dict,
    model_step: int,
    lead_time: "LEAD_TIME",
) -> Qube:
    """Create a Qube object from elements from model metadata and lead time."""
    surface_variables = {variables_metadata[var] for var in variables if variables_metadata[var].is_surface_level}
    pressure_variables = {variables_metadata[var] for var in variables if variables_metadata[var].is_pressure_level}
    model_variables = {variables_metadata[var] for var in variables if variables_metadata[var].is_model_level}

    lead_time_seconds = frequency_to_seconds(lead_time)
    steps = list(
        map(
            lambda x: x // 3600,
            range(model_step, lead_time_seconds + model_step, model_step),
        )
    )

    def make_qubes(objs: list[dict[str, Sequence]], metadata: dict[str, Any]) -> Qube:
        if not objs:
            return Qube.empty()

        qubes = [Qube.from_datacube(obj) for obj in objs]
        combined_qube: Qube = functools.reduce(operator.or_, qubes)  # type: ignore[reportArgumentType]
        combined_qube.add_metadata(metadata)
        return combined_qube

    pressure_qube = make_qubes(
        [
            {
                "step": steps,
                "levtype": "pl",
                "param": var.param,
                "level": var.level,
            }
            for var in pressure_variables
        ],
        {"name": "pressure"},
    )

    model_qube = make_qubes(
        [
            {
                "step": steps,
                "levtype": "ml",
                "param": var.param,
                "level": var.level,
            }
            for var in model_variables
        ],
        {"name": "model"},
    )

    surface_qube = make_qubes(
        [
            {
                "step": steps,
                "levtype": "sfc",
                "param": var.param,
            }
            for var in surface_variables
        ],
        {"name": "surface"},
    )

    return pressure_qube | model_qube | surface_qube


def parse_ensemble_members(
    ensemble_members: "ENSEMBLE_MEMBER_SPECIFICATION | None",
) -> list[int] | list[None]:
    """Parse an ensemble member specification into a list of member ids.

    Parameters
    ----------
    ensemble_members : ENSEMBLE_MEMBER_SPECIFICATION | None
        None gives no ensemble dimension and returns ``[None]``.
        An int is a single member id, e.g. ``3`` gives ``[3]``, and ``0`` is the
        control member. It is an id, not a count: use ``range(1, n + 1)``
        for ``n`` members. A sequence of int (e.g. ``range(1, 51)``) gives
        those member ids.

    Returns
    -------
    list[int] | list[None]
        The member ids.

    Raises
    ------
    ValueError
        If an int member id is negative.
    """
    if ensemble_members is None:
        return [None]
    if isinstance(ensemble_members, int):
        if ensemble_members < 0:
            raise ValueError("Ensemble member id must not be negative.")
        return [ensemble_members]
    return list(ensemble_members)


def expose_ensemble_dimension(x: dict | None, ens_mem: int | None) -> dict | None:
    """Label each dataset state in `x` with the ensemble member.

    The input is not modified, as it may be shared between branches
    labelled with different members. A new outer dict and new per-dataset
    dicts are returned (shallow copies, the arrays are not copied).

    A `None` input is a graph placeholder for initial conditions that are
    supplied elsewhere. It is returned unchanged, as there is no state to
    label.
    """
    if x is None:
        return None
    assert isinstance(x, dict), "Input state must be a dictionary"
    result: dict = {}
    for key, value in x.items():
        if isinstance(value, dict):
            value = dict(value)
            if ens_mem is not None:
                value[ENSEMBLE_DIMENSION_NAME] = ens_mem
        result[key] = value
    return result


def faked_ensemble_transform(act: fluent.Action, ens_num: int | None = None) -> fluent.Action:
    """Transform the action to simulate ensemble members"""
    return act.map(fluent.create_task_instance(expose_ensemble_dimension, [fluent.Node.Index(0), ens_num]))


def _add_self_to_environment(environment: E) -> E:
    """
    Add earthkit-workflows-anemoi to the environment list.

    The input is never modified, a new list (or dict of new lists) is returned.

    Parameters
    ----------
    environment : list[str] | dict[str, list[str]]
        Environment list, or dict of environment lists, to add self to

    Returns
    -------
    list[str] | dict[str, list[str]]
        New environment list with self added
    """
    from . import __version__ as version

    def copy_environment(env: Any) -> Any:
        if isinstance(env, list):
            return list(env)
        if isinstance(env, dict):
            return {key: list(value) for key, value in env.items()}
        return env

    if version.split(".")[-1].startswith("dev"):
        # If the version is a development version, ignore it
        # such that it is not overwritten
        # e.g. "0.3.1.dev0" -> "0.3"
        return copy_environment(environment)

    version = ".".join(version.split(".")[:3])  # Ensure version is in x.y.z format

    package_name = "earthkit-workflows-anemoi"
    self_var = f"{package_name}~={version}"

    def add_self_to_list(env_list: list[str]) -> list[str]:
        if len(env_list) == 0:
            # If the environment is empty, leave it as such
            return []

        if any(str(e).startswith(package_name) for e in env_list):
            # If the environment already contains the self variable, return it as is
            return list(env_list)

        return [*env_list, self_var]

    if isinstance(environment, list):
        return add_self_to_list(environment)
    if isinstance(environment, dict):
        return {key: add_self_to_list(value) for key, value in environment.items()}
    return environment


def _split_environment(environment: "ENVIRONMENT | None", keys: Sequence[str]) -> dict[str, list[str]]:
    """Split the environment into a dictionary of new lists, one per key, without adding self."""
    if environment is None:
        return {k: [] for k in keys}
    elif isinstance(environment, list):
        return {k: list(environment) for k in keys}
    elif isinstance(environment, dict):
        return {k: list(environment.get(k, [])) for k in keys}
    else:
        raise TypeError(f"Invalid type for environment: {type(environment)}. Must be list or dict.")


def crack_environment(environment: ENVIRONMENT | None, keys: list[str]) -> dict[str, list[str]]:
    """Crack the environment into a dictionary of lists.

    Every returned list is a new object, the input is never modified and no list is shared
    between keys. Self is added to each non-empty list, see `_add_self_to_environment`.
    """
    return _add_self_to_environment(_split_environment(environment, keys))


def warn_environment_deprecated(func_name: str, stacklevel: int = 3) -> None:
    """Warn that the `environment` argument is deprecated in favour of `requirements`.

    Parameters
    ----------
    func_name : str
        Name of the public function or method that received `environment`.
    stacklevel : int, optional
        Passed on to `warnings.warn`. With the default of 3 the warning is attributed to the
        caller of the function that calls this helper, i.e. the user's line.
    """
    warnings.warn(
        f"The `environment` argument of `{func_name}` is deprecated and will be removed in a future release. "
        "Use `requirements=Requirements(environment=...)`, or a dict of requirements keyed by stage "
        f"({', '.join(STAGES)}), instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def _as_requirements(value: "Requirements | Mapping[str, Any] | None", what: str) -> Requirements:
    """Return a new `Requirements` from a `Requirements`, a mapping of its fields or None."""
    if value is None:
        return Requirements()
    if isinstance(value, Requirements):
        return Requirements(environment=list(value.environment), needs_gpu=value.needs_gpu)
    if isinstance(value, Mapping):
        return Requirements(**value)
    raise TypeError(f"Invalid type for {what}: {type(value)}. Must be Requirements or a dict of its fields.")


def _merge_requirements(into: Requirements, other: Requirements) -> None:
    """Merge `other` into `into` in place, matching `earthkit.workflows.metadata.update_requirements`.

    The environment is the union of both (order is not significant) and `needs_gpu` is `other or into`.
    """
    into.environment = list(dict.fromkeys([*into.environment, *other.environment]))
    into.needs_gpu = other.needs_gpu or into.needs_gpu


def crack_requirements(
    requirements: "REQUIREMENTS | None",
    keys: Sequence[str],
    *,
    environment: "ENVIRONMENT | None" = None,
) -> dict[str, Requirements]:
    """Normalise execution requirements into one `Requirements` per stage.

    Parameters
    ----------
    requirements : REQUIREMENTS | None
        One of

        - None, meaning no requirements.
        - A `Requirements`, or a dict of its fields (``environment``, ``needs_gpu``).
          It is applied to every stage in `keys`.
        - A mapping keyed by stage (see `STAGES`), with `Requirements` or dicts of its
          fields as values. Stages missing from the mapping get empty requirements.
          Valid stages not in `keys` are ignored.

        An empty dict is treated as no requirements.
    keys : Sequence[str]
        The stages to return requirements for.
    environment : list[str] | dict[str, list[str]] | None, optional
        Deprecated, use `requirements`. A list is applied to every stage in `keys`, a dict is
        keyed by stage. It is merged into `requirements`.

    Returns
    -------
    dict[str, Requirements]
        New `Requirements` for each key in `keys`. No object is shared between keys or with
        the inputs, and the inputs are never modified. ``earthkit-workflows-anemoi`` is added
        to the environment of each stage whose environment is not empty, unless it already
        contains an ``earthkit-workflows-anemoi`` entry.

    Raises
    ------
    ValueError
        If `requirements` is a dict with unknown keys, or keys that mix `Requirements` fields
        and stage names, or stage names not in `STAGES`.
    TypeError
        If `requirements`, one of its values, or `environment` has an invalid type.

    Notes
    -----
    Requirements are merged additively, as `earthkit.workflows.metadata.update_requirements`
    does for nodes: the environment is the union of all packages (order is not guaranteed) and
    ``needs_gpu`` is true if it is true in any of them. ``False`` cannot override ``True``.
    """
    fields = set(Requirements.model_fields)
    result: dict[str, Requirements]

    if requirements is None:
        result = {k: Requirements() for k in keys}
    elif isinstance(requirements, Requirements):
        result = {k: _as_requirements(requirements, "requirements") for k in keys}
    elif isinstance(requirements, Mapping):
        given = set(requirements)
        if not given:
            result = {k: Requirements() for k in keys}
        elif given <= fields:
            result = {k: _as_requirements(requirements, "requirements") for k in keys}
        elif given <= set(STAGES):
            result = {k: _as_requirements(requirements.get(k), f"requirements['{k}']") for k in keys}
        else:
            raise ValueError(
                f"Invalid keys in requirements: {sorted(map(str, given))}. Keys must either all be "
                f"Requirements fields {sorted(fields)}, to apply to every stage, or all be stages "
                f"{list(STAGES)}, to apply per stage. Mixing them is not allowed."
            )
    else:
        raise TypeError(
            f"Invalid type for requirements: {type(requirements)}. Must be Requirements or a dict of "
            "its fields, or a dict of those keyed by stage."
        )

    for k, env in _split_environment(environment, keys).items():
        _merge_requirements(result[k], Requirements(environment=env))

    for req in result.values():
        req.environment = _add_self_to_environment(req.environment)

    return result
