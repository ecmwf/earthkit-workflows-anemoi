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
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Union

VALID_CKPT = Union[os.PathLike, str, dict[str, Any]]
LEAD_TIME = Union[int, str, datetime.timedelta]
DATE = Union[datetime.datetime, tuple[int, int, int], str]
ENVIRONMENT = Optional[Union[dict[str, list[str]], list[str]]]

ENSEMBLE_MEMBER_SPECIFICATION = Union[int, Sequence[int], None]
ENSEMBLE_DIMENSION_NAME: str = "ensemble_member"
