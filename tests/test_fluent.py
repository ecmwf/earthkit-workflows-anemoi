# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from anemoi.inference.testing import fake_checkpoints
from xarray import DataArray, DataTree

from earthkit.workflows.plugins.anemoi.fluent import Action, from_config, from_initial_conditions, from_input
from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

STANDARD_INFERENCE_TESTS = [
    # Test inputs of ensembles
    [
        "simple",
        2,
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 2, "param": 6, "date": 1},
    ],
    [
        "simple",
        8,
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 8, "param": 6, "date": 1},
    ],
    [
        "simple",
        8,
        {"date": "2020-01-01", "lead_time": "4D"},
        {"step": 16, ENSEMBLE_DIMENSION_NAME: 8, "param": 6, "date": 1},
    ],
    # Test different model configs
    [
        "simple",
        2,
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 2, "param": 6, "date": 1},
    ],
    [
        "surface",
        2,
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 2, "param": 6, "date": 1},
    ],
    [
        "full_atmo",
        2,
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 2, "param": 6, "date": 1, "level": 1},
    ],
]


def assert_shape(action: Action, shape: dict[str, int]):
    """Assert action nodes are of the correct shape"""
    shapes = defaultdict(set)

    def count(node: DataTree | DataArray):
        for dim, size in node.coords.items():
            shapes[dim].update(np.atleast_1d(size.values))

        if isinstance(node, DataTree):
            for child in node.children:
                count(node[child])

    count(action.nodes)

    for dim in shape:
        assert dim in shapes
        assert len(shapes[dim]) == shape[dim]


@pytest.mark.parametrize("ckpt, ensemble_members, kwargs, shape", STANDARD_INFERENCE_TESTS)
@fake_checkpoints
def test_from_input(ckpt, ensemble_members, kwargs, shape):
    """Test running from initial conditions"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()

    action = from_input(ckpt_full_path, "dummy", ensemble_members=ensemble_members, **kwargs)
    assert_shape(action, shape)


@pytest.mark.parametrize(
    "ckpt, ensemble_members, kwargs, shape",
    STANDARD_INFERENCE_TESTS,
)
@fake_checkpoints
def test_from_config(mock_config, ckpt, ensemble_members, kwargs, shape):
    """Test running from initial conditions"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()

    action = from_config(
        mock_config, ensemble_members=ensemble_members, **kwargs, checkpoint=str(ckpt_full_path), input="dummy"
    )
    assert_shape(action, shape)


@pytest.mark.parametrize(
    "ckpt, ensemble_members, kwargs, shape",
    STANDARD_INFERENCE_TESTS,
)
@fake_checkpoints
def test_from_initial_conditions_from_none(ckpt, ensemble_members, kwargs, shape):
    """Test running from initial conditions"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()
    kwargs.pop("date", None)

    action = from_initial_conditions(ckpt_full_path, None, ensemble_members=ensemble_members, **kwargs)
    shape.pop("date", None)
    assert_shape(action, shape)


@pytest.mark.parametrize(
    "ckpt, ensemble_members, kwargs, shape",
    STANDARD_INFERENCE_TESTS,
)
@fake_checkpoints
def test_from_initial_conditions_with_no_checkpoint_file(ckpt, ensemble_members, kwargs, shape):
    """Test running with no checkpoint file"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()

    from anemoi.inference.checkpoint import Checkpoint

    metadata = Checkpoint(ckpt_full_path)._metadata  # type: ignore

    kwargs.pop("date", None)

    action = from_initial_conditions(
        "non_existent_checkpoint.ckpt", None, ensemble_members=ensemble_members, metadata=metadata, **kwargs
    )
    shape.pop("date", None)
    assert_shape(action, shape)


@pytest.mark.parametrize(
    "ckpt, ensemble_members, kwargs, shape",
    STANDARD_INFERENCE_TESTS,
)
@fake_checkpoints
def test_from_initial_conditions_from_action(ckpt, ensemble_members, kwargs, shape):
    """Test running from initial conditions"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()
    kwargs.pop("date", None)

    from earthkit.workflows import fluent

    init_conditions = fluent.from_source(
        [None for _ in range(ensemble_members)],
        dims=[ENSEMBLE_DIMENSION_NAME],
        coords={ENSEMBLE_DIMENSION_NAME: range(ensemble_members)},
    )
    shape.pop("date", None)

    action = from_initial_conditions(ckpt_full_path, init_conditions, **kwargs)
    assert_shape(action, shape)


@pytest.mark.parametrize(
    "ckpt, ensemble_members, kwargs, shape",
    STANDARD_INFERENCE_TESTS,
)
@fake_checkpoints
def test_from_initial_conditions_from_infer(ckpt, ensemble_members, kwargs, shape):
    """Test running from initial conditions"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()
    kwargs.pop("date", None)

    from earthkit.workflows import fluent

    init_conditions = Action(
        fluent.from_source(
            [None for _ in range(ensemble_members)],
            dims=[ENSEMBLE_DIMENSION_NAME],
            coords={ENSEMBLE_DIMENSION_NAME: range(ensemble_members)},
        ).nodes
    )
    shape.pop("date", None)

    action = init_conditions.infer(ckpt_full_path, **kwargs)
    assert_shape(action, shape)
