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
        """The mocked IC function should return a dict with required keys."""
        ic_func = mock_registry["earthkit.workflows.plugins.anemoi.inference._get_initial_conditions"]
        state = ic_func(config={}, date="2020-01-01")
        assert isinstance(state, dict)
        assert "date" in state
        assert "fields" in state
        assert "latitudes" in state
        assert "longitudes" in state
        assert ENSEMBLE_DIMENSION_NAME not in state  # deterministic

        state_ens = ic_func(config={}, date="2020-01-01", number=3)
        assert ENSEMBLE_DIMENSION_NAME in state_ens
        assert state_ens[ENSEMBLE_DIMENSION_NAME] == 3

    @fake_checkpoints
    def test_run_mock_yields_fieldlists(self, simple_ckpt_path, mock_registry):
        """The mocked run function should yield fieldlist-like objects."""
        ic_func = mock_registry["earthkit.workflows.plugins.anemoi.inference._get_initial_conditions"]
        run_func = mock_registry["earthkit.workflows.plugins.anemoi.inference.run_as_earthkit"]

        state = ic_func(config={}, date="2020-01-01")
        results = list(run_func(state, config={}, lead_time=datetime.timedelta(days=1)))
        assert len(results) == 4  # 4 x 6h steps in 1D
        for r in results:
            assert hasattr(r, "fields")
            assert len(r.fields) > 0
