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
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any
from typing import Self

import numpy as np
from anemoi.utils.dates import frequency_to_seconds
from earthkit.workflows.fluent import Payload
from qubed import Qube

if TYPE_CHECKING:
    from anemoi.inference.metadata import Metadata
    from earthkit.workflows.fluent import Action


def _convert_num_to_abc(num: int) -> str:
    """Convert a number to its corresponding alphabetical representation.
    0 -> 'a', 1 -> 'b', ..., 25 -> 'z', 26 -> 'aa', etc.
    """
    result = ""
    while True:
        num, remainder = divmod(num, 26)
        result = chr(97 + remainder) + result
        if num == 0:
            break
        num -= 1
    return result


@dataclass
class Expansion:
    """Utilise Qubed to expand actions along the qube's axes.

    The Expansion class provides a mechanism for expanding workflow actions
    across multiple dimensions using the Qube data structure.

    The expansion process:

    1. Processes each dimension in the qube structure
    2. Expands actions along each axis sequentially
    3. Splits actions into separate branches when multiple child dimensions exist
    4. Creates a hierarchical structure organised by metadata names

    Parameters
    ----------
    qube : Qube
        A Qube object representing the multi-dimensional structure to expand over.
        The qube defines the axes (dimensions) and their values that will be used
        to expand the action.

    Examples
    --------
    Create an expansion for surface and pressure level variables:

    >>> from qubed import Qube
    >>> from earthkit.workflows.plugins.anemoi.utils import Expansion
    >>>
    >>> # Create surface variables qube
    >>> surface = Qube.from_datacube({
    ...     "step": [6, 12, 18],
    ...     "param": ["2t", "10u", "10v"]
    ... })
    >>> surface.add_metadata({"name": "surface"})
    >>>
    >>> # Create pressure level variables qube
    >>> pressure = Qube.from_datacube({
    ...     "step": [6, 12, 18],
    ...     "param": ["t", "u", "v"],
    ...     "level": [500, 850, 1000]
    ... })
    >>> pressure.add_metadata({"name": "pressure"})
    >>>
    >>> # Combine into hierarchical structure
    >>> combined = surface | pressure
    >>> expansion = Expansion(combined)
    >>>
    >>> # Expand an action
    >>> expanded_action = expansion.expand(action)

    See Also
    --------
    expansion_coordinates : Create an Expansion from model metadata
    """

    qube: Qube

    def axes(self) -> dict[str, set[str]]:
        """Return a dictionary of all the axes and their values in the qube.

        Returns
        -------
        dict[str, set[str]]
            Dictionary mapping axis names to sets of values for that axis.

        Examples
        --------
        >>> qube = Qube.from_datacube({
        ...     "step": [6, 12],
        ...     "param": ["t", "q"]
        ... })
        >>> expansion = Expansion(qube)
        >>> expansion.axes()
        {'step': {6, 12}, 'param': {'t', 'q'}}
        """
        return self.qube.axes()

    def drop_axis(self, axis: str | list[str]) -> Self:
        """Drop one or more axes from the qube.

        This creates a new Expansion instance with the specified axis/axes removed
        from the qube structure. The original Expansion is not modified.

        Parameters
        ----------
        axis : str | list[str]
            The name(s) of the axis/axes to drop from the qube.

        Returns
        -------
        Self
            A new Expansion instance with the specified axis/axes removed.

        Examples
        --------
        Drop a single axis:

        >>> qube = Qube.from_datacube({
        ...     "step": [6, 12],
        ...     "param": ["t", "q"],
        ...     "level": [500, 850]
        ... })
        >>> expansion = Expansion(qube)
        >>> new_expansion = expansion.drop_axis("level")
        >>> "level" in new_expansion.axes()
        False

        Drop multiple axes:

        >>> new_expansion = expansion.drop_axis(["step", "level"])
        >>> list(new_expansion.axes().keys())
        ['param']
        """
        new_qube = self.qube.remove_by_key(axis)
        return self.__class__(new_qube)

    def expand(self, action: "Action") -> "Action":
        """Expand the action according to the qube structure.

        This method recursively expands an action across all dimensions defined
        in the qube. For qubes with a single child, it expands sequentially through
        the dimensions. For qubes with multiple children, it splits the action into
        separate branches, each named according to the child's metadata (if present)
        or using alphabetical naming as a fallback.

        Parameters
        ----------
        action : Action
            The workflow action to expand across the qube's dimensions.

        Returns
        -------
        Action
            The expanded action with all dimensions applied. The action will have
            a hierarchical structure if the qube has multiple children.

        Notes
        -----
        The expansion algorithm works as follows:

        1. If the qube has no children, the action is returned unchanged
        2. The action is expanded along the first child's dimensions
        3. If multiple children exist, the action is split into branches
        4. Each branch is recursively expanded with its child's dimensions
        5. Child branches are named using metadata or alphabetical fallback

        Examples
        --------
        Simple single-dimension expansion:

        >>> qube = Qube.from_datacube({"step": [6, 12, 18]})
        >>> expansion = Expansion(qube)
        >>> expanded = expansion.expand(action)
        # Action is now expanded over step=[6, 12, 18]

        Hierarchical expansion with surface and pressure levels:

        >>> # Qube structure:
        >>> # root, step=6/12
        >>> # ├── param=2t/10u/10v (surface)
        >>> # └── param=t/u/v, level=500/850/1000 (pressure)
        >>>
        >>> expansion = Expansion(qube)
        >>> expanded = expansion.expand(action)
        # Action is expanded over step, then split into /surface and /pressure
        # branches, each with their respective param and level dimensions

        Drop an axis before expansion:

        >>> expansion = Expansion(qube)
        >>> no_step = expansion.drop_axis("step")
        >>> expanded = no_step.expand(action)
        # Action expanded over remaining dimensions only

        See Also
        --------
        drop_axis : Remove axes before expansion
        axes : View available axes in the qube
        """

        def select(x, key, val):
            return x.sel(**{key: val})

        def get_name(child: Qube, index: int) -> str:
            if "name" in child.metadata:
                name_meta = child.metadata["name"]
                return str(np.unique_values(name_meta).flatten()[0])
            return _convert_num_to_abc(index)

        def expand_fn(action: "Action", qube: Qube, path: str) -> "Action":
            """Recursively expand the action based on the qube structure."""
            if not qube.key == "root":  # Skip the root key
                # Expand along the current qube's key and values
                action = action.expand((qube.key, list(qube.values)), (qube.key, list(qube.values)), path=path)

            num_children = len(qube.children)
            if num_children == 0:  # Base case: no more children to expand
                return action

            if num_children == 1:  # In the case of one child, no need to split, just continue expanding
                return expand_fn(action, qube.children[0], path)

            action = action.split(
                {
                    f"{path}/{get_name(child, i)}": Payload(
                        select, kwargs={"key": child.key, "val": list(child.values)}
                    )
                    for i, child in enumerate(qube.children)
                }
            )
            for i, child in enumerate(qube.children):
                sub_path = f"{path}/{get_name(child, i)}"
                action = expand_fn(action, child, sub_path)

            return action

        if not self.qube.children:
            return action

        return expand_fn(action, self.qube, "")


def expansion_coordinates(metadata: "Metadata", lead_time: int | str | timedelta) -> Expansion:
    """Create an Expansion object from model metadata and lead time.

    This function constructs an Expansion object by analysing the model's metadata
    to identify surface, pressure level, and model level variables. It creates a
    hierarchical qube structure organising variables by their vertical coordinate
    type and expands them across time steps.

    Parameters
    ----------
    metadata : Metadata
        Model metadata containing variable definitions, including their vertical
        coordinate information (surface, pressure levels, model levels) and the
        model's time step.
    lead_time : int | str | timedelta
        The forecast lead time as an integer or string (e.g., "7D" for 7 days).
        This determines the number of time steps in the expansion.
        If an integer is provided, it is interpreted as hours.

    Returns
    -------
    Expansion
        An Expansion object with a hierarchical qube structure containing three
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
    >>> expansion = expansion_coordinates(metadata, "5D")
    >>> expansion.axes()
    {'step': {6, 12, 18, ..., 120}, 'param': {...}, 'level': {...}}

    Use with an action to expand across all dimensions:

    >>> expanded_action = expansion.expand(action)

    See Also
    --------
    Expansion : The Expansion class for manual qube construction
    """
    variables = metadata.select_variables(include=["diagnostic", "prognostic"], has_mars_requests=False)
    variables_metadata = metadata.typed_variables

    surface_variables = {variables_metadata[var] for var in variables if variables_metadata[var].is_surface_level}
    pressure_variables = {variables_metadata[var] for var in variables if variables_metadata[var].is_pressure_level}
    model_variables = {variables_metadata[var] for var in variables if variables_metadata[var].is_model_level}

    model_step = metadata.timestep.seconds
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

    return Expansion(pressure_qube | model_qube | surface_qube)
