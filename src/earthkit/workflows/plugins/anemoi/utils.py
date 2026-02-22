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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from anemoi.utils.dates import frequency_to_seconds
from qubed import Qube

from earthkit.workflows import fluent

from .types import ENVIRONMENT

if TYPE_CHECKING:
    from anemoi.inference.metadata import Metadata

    from .types import ENSEMBLE_MEMBER_SPECIFICATION, ENVIRONMENT, LEAD_TIME

E = TypeVar("E", bound=ENVIRONMENT)


def expansion_qube(metadata: "Metadata", lead_time: "LEAD_TIME") -> Qube:
    """Create a Qube object from model metadata and lead time.

    This function constructs a Qube object by analysing the model's metadata
    to identify surface, pressure level, and model level variables. It creates a
    hierarchical qube structure organising variables by their vertical coordinate
    type and expands them across time steps.

    Parameters
    ----------
    metadata : Metadata
        Model metadata containing variable definitions, including their vertical
        coordinate information (surface, pressure levels, model levels) and the
        model's time step.
    lead_time : LEAD_TIME
        The forecast lead time as an integer or string (e.g., "7D" for 7 days).
        This determines the number of time steps in the expansion.
        If an integer is provided, it is interpreted as hours.

    Returns
    -------
    Qube
        A qube object with a hierarchical structure containing up to three
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
    variables = metadata.select_variables(include=["diagnostic", "prognostic"], has_mars_requests=False)
    variables_metadata = metadata.typed_variables
    model_step = metadata.timestep.seconds
    return _expansion_qube(variables, variables_metadata, model_step, lead_time)


def _expansion_qube(variables: list[str], variables_metadata: dict, model_step: int, lead_time: "LEAD_TIME") -> Qube:
    """Create a Qube object from elements from model metadata and lead time."""
    surface_variables = {variables_metadata[var] for var in variables if variables_metadata[var].is_surface_level}
    pressure_variables = {variables_metadata[var] for var in variables if variables_metadata[var].is_pressure_level}
    model_variables = {variables_metadata[var] for var in variables if variables_metadata[var].is_model_level}

    lead_time_seconds = frequency_to_seconds(lead_time)
    steps = list(map(lambda x: x // 3600, range(model_step, lead_time_seconds + model_step, model_step)))

    def make_qubes(objs: list[dict[str, Sequence]], metadata: dict[str, Any]) -> Qube:
        if not objs:
            return Qube.empty()

        qubes = [Qube.from_datacube(obj) for obj in objs]
        combined_qube: Qube = functools.reduce(operator.or_, qubes)  # pyright: ignore[reportArgumentType]
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


def parse_ensemble_members(ensemble_members: "ENSEMBLE_MEMBER_SPECIFICATION | None") -> list[int] | list[None]:
    """Parse ensemble members"""
    if ensemble_members is None:
        return [None]
    if isinstance(ensemble_members, int):
        if ensemble_members < 1:
            raise ValueError("Number of ensemble members must be greater than 0.")
        return list(range(1, ensemble_members + 1))
    return list(ensemble_members)


def faked_ensemble_transform(act: fluent.Action, ens_num: int | None = None) -> fluent.Action:
    """Transform the action to simulate ensemble members"""

    def _empty_payload(x, ens_mem: int | None):
        assert isinstance(x, dict), "Input state must be a dictionary"
        if ens_mem is not None:
            x["ensemble_member"] = ens_mem
        return x

    return act.map(fluent.Payload(_empty_payload, [fluent.Node.input_name(0), ens_num]))


def _add_self_to_environment(environment: E) -> E:
    """
    Add earthkit-workflows-anemoi to the environment list.

    Parameters
    ----------
    environment : list[str] | dict[str, list[str]]
        Environment list to self in place to

    Returns
    -------
    list[str] | dict[str, list[str]]
        Environment list with self added
    """
    from . import __version__ as version

    if version.split(".")[-1].startswith("dev"):
        # If the version is a development version, ignore it
        # such that it is not overwritten
        # e.g. "0.3.1.dev0" -> "0.3"
        return environment

    version = ".".join(version.split(".")[:3])  # Ensure version is in x.y.z format

    package_name = "earthkit-workflows-anemoi"
    self_var = f"{package_name}~={version}"

    def add_self_to_list(env_list: list[str]) -> list[str]:
        if len(env_list) == 0:
            # If the environment is empty, leave it as such
            return []

        if any(str(e).startswith(package_name) for e in env_list):
            # If the environment already contains the self variable, return it as is
            return env_list

        env_list.append(self_var)
        return env_list

    if isinstance(environment, list):
        environment = add_self_to_list(environment)
    elif isinstance(environment, dict):
        for key in environment:
            environment[key] = add_self_to_list(environment[key])
    return environment


def crack_environment(environment: ENVIRONMENT | None, keys: list[str]) -> dict[str, list[str]]:
    """Crack the environment into a dictionary of lists."""
    if environment is None:
        return _add_self_to_environment({k: [] for k in keys})
    elif isinstance(environment, list):
        return _add_self_to_environment({k: environment for k in keys})
    elif isinstance(environment, dict):
        return _add_self_to_environment({k: environment.get(k, []) for k in keys})
    else:
        raise TypeError(f"Invalid type for environment: {type(environment)}. Must be list or dict.")
