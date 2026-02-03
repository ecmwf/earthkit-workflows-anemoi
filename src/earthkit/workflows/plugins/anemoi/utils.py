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
from typing import TYPE_CHECKING
from typing import Any

from anemoi.utils.dates import frequency_to_seconds
from qubed import Qube

if TYPE_CHECKING:
    from anemoi.inference.metadata import Metadata

    from earthkit.workflows.plugins.anemoi.types import LEAD_TIME


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

    return pressure_qube | model_qube | surface_qube
