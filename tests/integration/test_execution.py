# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Execution tests with mocked anemoi inference functions.

These tests verify that the anemoi payload functions (IC fetch and model run)
can be resolved and called with correct arguments. Internal earthkit-workflows
operations (Backend.take etc.) are skipped as they need the full Cascade runtime.
"""

from __future__ import annotations

import datetime

import pytest
from anemoi.inference.testing import fake_checkpoints

from earthkit.workflows.plugins.anemoi.fluent import from_input
from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

pytestmark = pytest.mark.integration


class TestExecution:
    """Execute anemoi-specific payloads from the DAG with mocked inference functions."""

    @fake_checkpoints
    def test_anemoi_payloads_callable(self, simple_ckpt_path, mock_registry):
        """Verify the anemoi payload functions exist in mock registry and are callable."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        graph = action.graph()

        anemoi_funcs_found = set()
        for node in graph.nodes():
            p = node.payload
            if p is None or not hasattr(p, "func"):
                continue
            if not isinstance(p.func, str):
                continue

            func_path = p.func
            anemoi_funcs_found.add(func_path)
            # Verify the mock exists
            assert func_path in mock_registry, f"No mock registered for {func_path}"
            assert callable(mock_registry[func_path])

        # Should have found at least the IC and model run functions
        assert any("initial_conditions" in f for f in anemoi_funcs_found)
        assert any("run_as_earthkit" in f for f in anemoi_funcs_found)

    @fake_checkpoints
    def test_ic_mock_returns_valid_state(self, simple_ckpt_path, mock_registry):
        """The mocked IC function should return dict[str, State] (multi-dataset API)."""
        ic_func = mock_registry["earthkit.workflows.plugins.anemoi.inference._get_initial_conditions"]
        state_dict = ic_func(config={}, date="2020-01-01")

        # New API returns dict[str, State]
        assert isinstance(state_dict, dict)
        assert "era5" in state_dict  # default dataset name in mock

        state = state_dict["era5"]
        assert isinstance(state, dict)
        assert "date" in state
        assert "fields" in state
        assert "latitudes" in state
        assert "longitudes" in state
        assert ENSEMBLE_DIMENSION_NAME not in state  # deterministic

        state_dict_ens = ic_func(config={}, date="2020-01-01", number=3)
        state_ens = state_dict_ens["era5"]
        assert ENSEMBLE_DIMENSION_NAME in state_ens
        assert state_ens[ENSEMBLE_DIMENSION_NAME] == 3

    @fake_checkpoints
    def test_run_mock_yields_fieldlists(self, simple_ckpt_path, mock_registry):
        """The mocked run function should yield dict[str, SimpleFieldList] per step."""
        ic_func = mock_registry["earthkit.workflows.plugins.anemoi.inference._get_initial_conditions"]
        run_func = mock_registry["earthkit.workflows.plugins.anemoi.inference.run_as_earthkit"]

        state_dict = ic_func(config={}, date="2020-01-01")
        results = list(run_func(state_dict, config={}, lead_time=datetime.timedelta(days=1)))

        assert len(results) == 4  # 4 x 6h steps in 1D

        for step_result in results:
            # New API yields dict[str, SimpleFieldList]
            assert isinstance(step_result, dict)
            assert "era5" in step_result  # default dataset name in mock

            fieldlist = step_result["era5"]
            assert hasattr(fieldlist, "fields")
            assert len(fieldlist.fields) > 0
