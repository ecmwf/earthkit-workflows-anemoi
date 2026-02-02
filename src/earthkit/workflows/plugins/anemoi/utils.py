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


def convert_num_to_abc(num: int) -> str:
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
    """Utilise Qubed to expand actions along the qube's axes."""

    qube: Qube

    def axes(self) -> dict[str, set[str]]:
        """Return a dictionary of all the spans of the keys in the qube."""
        return self.qube.axes()

    def drop_axis(self, axis: str | list[str]) -> Self:
        """Drop an axis from the qube."""
        new_qube = self.qube.remove_by_key(axis)
        return self.__class__(new_qube)

    def expand(self, action: "Action") -> "Action":
        """Expand the action according to the qube.

        Example:
        --------
        If the `qube` has the following structure:

        ```
        root, step=6/12
        ├── param=100u/100v/10u/10v/2d/2t/cp/hcc/lcc/mcc/msl/r...
        └── param=q/t/u/v/w/z, level=50/100/150/200/250/300/400/500/600/700/850/900/950/1000
        ```

        The expansion will result in an action that is expanded first over the `step` axis,
        and then split over the following children, resulting in a tree structure named if the
        children have `name` metadata, otherwise named alphabetically:

        ```
        expansion.expand(action).nodes

        <xarray.DataTree>
        Group: /
        ├── Group: /surface
        │       Dimensions:   (param: 24, step: 2, dim_0: 1)
        │       Coordinates:
        │         * param     (param) object 192B '100u' '100v' '10u' '10v' ... 'tcc' 'tcw' 'tp'
        │         * step      (step) int64 16B 6 12
        │       Dimensions without coordinates: dim_0
        │       Data variables:
        │           nodeset0  (param, step, dim_0) object 384B Node take:0d90f11217f62b6a0cdb...
        └── Group: /pressure
                Dimensions:   (level: 13, param: 6, step: 2, dim_0: 1)
                Coordinates:
                * level     (level) int64 104B 50 100 150 200 250 300 ... 600 700 850 925 1000
                * param     (param) object 48B 'q' 't' 'u' 'v' 'w' 'z'
                * step      (step) int64 16B 6 12
                Dimensions without coordinates: dim_0
                Data variables:
                    nodeset1  (level, param, step, dim_0) object 1kB Node take:22351a18053046...
        ```

        """

        def select(x, key, val):
            return x.sel(**{key: val})

        def get_name(child: Qube, index: int) -> str:
            if "name" in child.metadata:
                name_meta = child.metadata["name"]
                return str(np.unique_values(name_meta).flatten()[0])
            return convert_num_to_abc(index)

        def expand_fn(action: "Action", qube: Qube, path: str) -> "Action":
            """Recursively expand the action based on the qube structure."""
            if not qube.key == "root":  # Skip the root key
                # Expand along the current qube's key and values
                action = action.expand((qube.key, list(qube.values)), (qube.key, list(qube.values)), path=path)

            num_children = len(qube.children)
            if num_children == 0:  # Base case: no more children to expand
                return action

            if num_children == 1:  # In the case of one child, no need to split, just continue expanding
                child = qube.children[0]
                return expand_fn(action, child, path)

            action = action.split(
                {
                    f"{path}/{get_name(child, i)}": Payload(select, kwargs={"key": child.key, "val": child.values})
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


def expansion_coordinates(metadata: "Metadata", lead_time: int) -> Expansion:
    """
    Create an Expansion object based on the provided metadata and lead time.
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
                "param": var.param,
            }
            for var in surface_variables
        ],
        {"name": "surface"},
    )

    return Expansion(pressure_qube | model_qube | surface_qube)
