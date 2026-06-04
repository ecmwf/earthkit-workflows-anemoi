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
        ["2000-01-01", [2], {"date": 1}],
        ["2000-01-01", range(1, 2), {"date": 1}],
        ["2000-01-01", 10, {"date": 1, ENSEMBLE_DIMENSION_NAME: 10}],
        ["2000-01-01", range(10), {"date": 1, ENSEMBLE_DIMENSION_NAME: 10}],
        ["2000-01-01", range(10, 20), {"date": 1, ENSEMBLE_DIMENSION_NAME: 10}],
        ["2000-01-01", 10, {"date": 1, ENSEMBLE_DIMENSION_NAME: 10}],
        ["2000-01-01", 51, {"date": 1, ENSEMBLE_DIMENSION_NAME: 51}],
        ["2000-01-01", 51, {"date": 1, ENSEMBLE_DIMENSION_NAME: 51}],
        [-1, 51, {"date": 1, ENSEMBLE_DIMENSION_NAME: 51}],
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
    "date, ensemble_members, shape",
    [
        ["2000-01-01", 0, {"date": 1, ENSEMBLE_DIMENSION_NAME: 10}],
        ["2000-01-01", -1, {"date": 1, ENSEMBLE_DIMENSION_NAME: 10}],
    ],
)
def test_get_initial_conditions_action_fail(mock_config, date, ensemble_members, shape):
    """Test failing to get initial conditions"""
    with pytest.raises(ValueError):
        _ = _get_initial_conditions_source(mock_config, date, ensemble_members)
