# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest
from anemoi.inference.input import Input


class MockInput(Input):
    """Mock Input class"""

    def __init__(self):
        pass

    def load_forcings(self, *, variables, dates):
        pass

    def create_input_state(self, *, date=None):
        pass


@pytest.fixture
def mocked_metadata():
    import json

    return json.load(open(Path(__file__).parent / "metadata.json", "r"))
