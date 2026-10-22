# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from earthkit.workflows.plugins.anemoi.fluent import _get_initial_conditions_source
from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME


@pytest.mark.parametrize(
    "date, ensemble_members, shape",
    [
        ["2000-01-01", 1, {"date": 1}],
        ["2000-01-01", 0, {"date": 1}],
        ["2000-01-01", 10, {"date": 1}],
        ["2000-01-01", [2], {"date": 1}],
        ["2000-01-01", range(1, 2), {"date": 1}],
        ["2000-01-01", range(1, 11), {"date": 1, ENSEMBLE_DIMENSION_NAME: 10}],
        ["2000-01-01", range(10), {"date": 1, ENSEMBLE_DIMENSION_NAME: 10}],
        ["2000-01-01", range(10, 20), {"date": 1, ENSEMBLE_DIMENSION_NAME: 10}],
        ["2000-01-01", range(1, 52), {"date": 1, ENSEMBLE_DIMENSION_NAME: 51}],
        [-1, range(1, 52), {"date": 1, ENSEMBLE_DIMENSION_NAME: 51}],
    ],
)
def test_get_initial_conditions_action(mock_config, date, ensemble_members, shape):
    """Test getting initial conditions"""
    action = _get_initial_conditions_source(
        mock_config,
        date,
        ensemble_members,
    )

    for dim in shape:
        assert dim in action.nodes.dims
        assert action.nodes.coords[dim].size == shape[dim]


@pytest.mark.parametrize(
    "ensemble_members, expected",
    [
        [3, [3]],
        [0, [0]],
        [[3], [3]],
        [range(2, 3), [2]],
        [[0, 3], [0, 3]],
        [range(1, 4), [1, 2, 3]],
    ],
)
@pytest.mark.parametrize("config_kind", ["dict", "action"])
def test_get_initial_conditions_keeps_number_dimension(ensemble_members, expected, config_kind):
    """`number` is a real dimension, including for single members, for both config branches."""
    from earthkit.workflows import fluent

    config = {"checkpoint": "ckpt"}
    if config_kind == "action":
        config = fluent.from_source([config], dims=["config"])

    action = _get_initial_conditions_source(config, "2000-01-01", ensemble_members)

    assert ENSEMBLE_DIMENSION_NAME in action.nodes.dims
    assert action.nodes.sizes[ENSEMBLE_DIMENSION_NAME] == len(expected)
    assert list(action.nodes.coords[ENSEMBLE_DIMENSION_NAME].values) == expected


@pytest.mark.parametrize("config_kind", ["dict", "action"])
def test_get_initial_conditions_none_is_deterministic(config_kind):
    """`None` members: one task per date called with `number=None` and no `number` dimension."""
    from earthkit.workflows import fluent
    from earthkit.workflows.plugins.anemoi import inference

    config = {"checkpoint": "ckpt"}
    if config_kind == "action":
        config = fluent.from_source([config], dims=["config"])

    action = _get_initial_conditions_source(config, "2000-01-01", None)

    assert ENSEMBLE_DIMENSION_NAME not in action.nodes.dims
    assert ENSEMBLE_DIMENSION_NAME not in action.nodes.coords
    assert action.nodes.sizes["date"] == 1

    ic_tasks = [
        node.payload for node in action.graph().nodes() if "number" in getattr(node.payload, "static_input_kw", {})
    ]
    assert len(ic_tasks) == 1
    assert ic_tasks[0].static_input_kw["number"] is None
    assert ic_tasks[0].static_input_kw["date"] == "2000-01-01"
    assert ic_tasks[0].definition.entrypoint == f"{inference.__name__}._get_initial_conditions"


@pytest.mark.parametrize(
    "date, ensemble_members, shape",
    [
        ["2000-01-01", -1, {"date": 1, ENSEMBLE_DIMENSION_NAME: 10}],
    ],
)
def test_get_initial_conditions_action_fail(mock_config, date, ensemble_members, shape):
    """Test failing to get initial conditions"""
    with pytest.raises(ValueError):
        _ = _get_initial_conditions_source(mock_config, date, ensemble_members)
