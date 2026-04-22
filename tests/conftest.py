# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing.mock_checkpoint import MockRunConfiguration


@pytest.fixture
@fake_checkpoints
def mock_config(tmp_path: Path):
    import yaml

    parent_dir = Path(__file__).parent
    config_path = parent_dir / "configs" / "simple.yaml"

    config_dict = yaml.safe_load(config_path.read_text())
    config_dict["checkpoint"] = f"{parent_dir}/{config_dict['checkpoint']}"

    with open(tmp_path / "simple.yaml", "w") as f:
        yaml.safe_dump(config_dict, f)

    tmp_path = tmp_path / "simple.yaml"

    return MockRunConfiguration.load(
        str((tmp_path).absolute()),
        overrides=dict(runner="testing", device="cpu", input="dummy"),
    )
