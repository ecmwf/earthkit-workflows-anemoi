# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests with different checkpoint configurations."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
from anemoi.inference.testing import fake_checkpoints
from earthkit.workflows.fluent import nodetree_arrays

from earthkit.workflows.plugins.anemoi.fluent import from_input
from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

pytestmark = pytest.mark.integration


class TestMultiCheckpoint:
    """Test with different checkpoint configurations."""

    @fake_checkpoints
    def test_full_atmo_checkpoint(self, full_atmo_ckpt_path):
        """full_atmo checkpoint should produce a DAG with level dimensions."""
        action = from_input(full_atmo_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        graph = action.graph()
        assert not graph.has_cycle()

        # full_atmo has pressure levels
        shapes = defaultdict(set)
        for _, narray in nodetree_arrays(action.nodes):
            for dim, size in narray.coords.items():
                shapes[dim].update(np.atleast_1d(size.values))
        assert "level" in shapes

    @fake_checkpoints
    def test_full_atmo_with_ensemble(self, full_atmo_ckpt_path):
        """full_atmo with ensemble should have both level and ensemble coords."""
        action = from_input(
            full_atmo_ckpt_path,
            "dummy",
            date="2020-01-01",
            lead_time="1D",
            ensemble_members=2,
        )
        # Collect all dims and coords from the tree (including children)
        all_dims = set()
        all_coords = set()
        for _, narray in nodetree_arrays(action.nodes):
            all_dims.update(narray.dims)
            all_coords.update(narray.coords.keys())
        assert ENSEMBLE_DIMENSION_NAME in all_dims
        # level appears as a coordinate (possibly scalar) on pressure level groups
        assert "level" in all_coords

    @fake_checkpoints
    def test_multiple_lead_times(self, simple_ckpt_path):
        """Different lead times should produce different step counts."""
        action_1d = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        action_4d = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="4D")

        steps_1d = set()
        steps_4d = set()
        for _, narray in nodetree_arrays(action_1d.nodes):
            if "step" in narray.coords:
                steps_1d.update(np.atleast_1d(narray.coords["step"].values))
        for _, narray in nodetree_arrays(action_4d.nodes):
            if "step" in narray.coords:
                steps_4d.update(np.atleast_1d(narray.coords["step"].values))

        assert len(steps_4d) > len(steps_1d)
