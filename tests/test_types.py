# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ENSEMBLE_DIMENSION_NAME constant and its consistent usage.

The dimension name was changed from 'ensemble_member' to 'number' in commit
7fd7969. These tests verify the value is correct and that it is used
consistently across the codebase rather than hardcoded strings.
"""

from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME


def test_ensemble_dimension_name_value():
    """ENSEMBLE_DIMENSION_NAME should be 'number'."""
    assert ENSEMBLE_DIMENSION_NAME == "number"


def test_ensemble_dimension_name_is_string():
    """ENSEMBLE_DIMENSION_NAME should be a str."""
    assert isinstance(ENSEMBLE_DIMENSION_NAME, str)


def test_ensemble_dimension_name_reexported_from_init():
    """The constant should be re-exported from the package __init__."""
    from earthkit.workflows.plugins.anemoi import ENSEMBLE_DIMENSION_NAME as reexported

    assert reexported == "number"


def test_expose_ensemble_dimension_uses_constant():
    """expose_ensemble_dimension should use ENSEMBLE_DIMENSION_NAME, not a hardcoded key."""
    from earthkit.workflows.plugins.anemoi.utils import expose_ensemble_dimension

    state = {"data": {"date": "2020-01-01"}}
    result = expose_ensemble_dimension(state, 3)
    assert ENSEMBLE_DIMENSION_NAME in result["data"]
    assert result["data"][ENSEMBLE_DIMENSION_NAME] == 3
    # Must NOT use the old hardcoded key
    if ENSEMBLE_DIMENSION_NAME != "ensemble_member":
        assert "ensemble_member" not in result["data"]


def test_expose_ensemble_dimension_none_member():
    """When ensemble number is None, no ensemble key should be added."""
    from earthkit.workflows.plugins.anemoi.utils import expose_ensemble_dimension

    state = {"data": {"date": "2020-01-01"}}
    result = expose_ensemble_dimension(state, None)
    assert ENSEMBLE_DIMENSION_NAME not in result["data"]


def test_expose_ensemble_dimension_does_not_mutate_input():
    """Members 0 and 3 applied to one shared state are labelled separately, the source is unchanged."""
    from earthkit.workflows.plugins.anemoi.utils import expose_ensemble_dimension

    array = object()
    state = {"data": {"date": "2020-01-01", "field": array}, "other": {"date": "2020-01-01"}}

    control = expose_ensemble_dimension(state, 0)
    perturbed = expose_ensemble_dimension(state, 3)

    assert control["data"][ENSEMBLE_DIMENSION_NAME] == 0
    assert control["other"][ENSEMBLE_DIMENSION_NAME] == 0
    assert perturbed["data"][ENSEMBLE_DIMENSION_NAME] == 3
    assert perturbed["other"][ENSEMBLE_DIMENSION_NAME] == 3

    # Source is unchanged
    assert state == {"data": {"date": "2020-01-01", "field": array}, "other": {"date": "2020-01-01"}}
    assert ENSEMBLE_DIMENSION_NAME not in state["data"]

    # New outer and per-dataset dicts, but the values are shared, not copied
    for result in (control, perturbed):
        assert result is not state
        assert result["data"] is not state["data"]
        assert result["other"] is not state["other"]
        assert result["data"]["field"] is array
