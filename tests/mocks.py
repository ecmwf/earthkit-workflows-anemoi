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
