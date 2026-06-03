# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests verifying ensemble dimension naming is consistent throughout the pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from anemoi.inference.testing import fake_checkpoints
from earthkit.workflows.fluent import nodetree_arrays

from earthkit.workflows import fluent
from earthkit.workflows.plugins.anemoi.fluent import from_initial_conditions, from_input
from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

pytestmark = pytest.mark.integration


class TestEnsembleDimensionConsistency:
    """Verify ensemble dimension naming is consistent throughout the pipeline."""

    @fake_checkpoints
    def test_dimension_name_in_action_coords(self, simple_ckpt_path):
        """The action's coords should use ENSEMBLE_DIMENSION_NAME, not 'ensemble_member'."""
        action = from_input(
            simple_ckpt_path,
            "dummy",
            date="2020-01-01",
            lead_time="1D",
            ensemble_members=3,
        )
        assert ENSEMBLE_DIMENSION_NAME in action.nodes.coords
        assert ENSEMBLE_DIMENSION_NAME == "number"
        if ENSEMBLE_DIMENSION_NAME != "ensemble_member":
            assert "ensemble_member" not in action.nodes.dims

    @fake_checkpoints
    def test_dimension_name_in_payload_args(self, simple_ckpt_path):
        """Payload dimension references should use the current constant value."""
        action = from_input(
            simple_ckpt_path,
            "dummy",
            date="2020-01-01",
            lead_time="1D",
            ensemble_members=3,
        )
        # Check that payloads reference the correct dimension name in their kwargs
        for _, narray in nodetree_arrays(action.nodes):
            for node in np.atleast_1d(narray.values).flatten():
                if hasattr(node, "payload") and node.payload is not None:
                    kwargs = node.payload.kwargs
                    # If there's a dimension argument, it should match
                    if "dim" in kwargs:
                        dim_val = kwargs["dim"]
                        if isinstance(dim_val, tuple) and len(dim_val) == 2:
                            assert dim_val[0] == ENSEMBLE_DIMENSION_NAME

    @fake_checkpoints
    def test_from_initial_conditions_action_uses_correct_dim(self, simple_ckpt_path):
        """from_initial_conditions with a pre-built action should match dimension name."""
        init = fluent.from_source(
            [None, None, None],
            dims=[ENSEMBLE_DIMENSION_NAME],
            coords={ENSEMBLE_DIMENSION_NAME: [1, 2, 3]},
        )
        action = from_initial_conditions(simple_ckpt_path, init, lead_time="1D")
        assert ENSEMBLE_DIMENSION_NAME in action.nodes.dims

    @fake_checkpoints
    def test_expose_ensemble_dimension_in_mock_execution(self, simple_ckpt_path, executor):
        """Verify the ensemble key in executed states uses the constant name."""
        from earthkit.workflows.plugins.anemoi.utils import expose_ensemble_dimension

        state = {"data": {"date": "2020-01-01"}}
        result = expose_ensemble_dimension(state, 5)
        assert ENSEMBLE_DIMENSION_NAME in result["data"]
        assert result["data"][ENSEMBLE_DIMENSION_NAME] == 5
