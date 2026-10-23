# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import os
from collections.abc import Mapping, Sequence
from typing import Any, Union

from earthkit.workflows.metadata import Requirements

VALID_CKPT = Union[os.PathLike[str], str, Mapping[str, Any]]
LEAD_TIME = Union[int, str, datetime.timedelta]
DATE = Union[datetime.datetime, tuple[int, int, int], str]
ENVIRONMENT = Union[Mapping[str, list[str]], list[str]]

STAGES: tuple[str, ...] = ("inference", "initial_conditions")
"""Stages of a workflow to which execution requirements can be assigned."""

REQUIREMENTS = Union[
    Requirements,
    Mapping[str, Union[Requirements, Mapping[str, Any]]],
    Mapping[str, Any],
]
"""Execution requirements.

Either a single `Requirements` (or a plain mapping of its fields), applied to every stage,
or a mapping keyed by stage name (see `STAGES`) whose values are `Requirements` or mappings of its fields.
"""

# An int is a single ensemble member id (0 is the control member), not a count;
# a sequence (e.g. range(1, 51)) selects several members.
ENSEMBLE_MEMBER_SPECIFICATION = Union[int, Sequence[int]]
ENSEMBLE_DIMENSION_NAME: str = "number"
