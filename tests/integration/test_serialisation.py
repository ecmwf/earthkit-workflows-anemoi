# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests verifying graphs can be serialised and deserialised."""

from __future__ import annotations

import pytest
from anemoi.inference.testing import fake_checkpoints

from earthkit.workflows import Cascade, serialise
from earthkit.workflows.plugins.anemoi.fluent import from_input

pytestmark = pytest.mark.integration


class TestSerialisation:
    """Verify graphs can be serialised and deserialised."""

    @fake_checkpoints
    def test_graph_serialises(self, simple_ckpt_path):
        """The graph should serialise to a dict without errors."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        graph = action.graph()
        data = serialise(graph)
        assert isinstance(data, dict)
        assert len(data) > 0

    @fake_checkpoints
    def test_cascade_from_actions(self, simple_ckpt_path):
        """Should be able to create a Cascade object from actions."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        cascade = Cascade.from_actions([action])
        assert cascade is not None

    @fake_checkpoints
    def test_cascade_serialise_round_trip(self, simple_ckpt_path, tmp_path):
        """Cascade serialise -> deserialise should not error."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        cascade = Cascade.from_actions([action])

        filepath = tmp_path / "test_cascade.pkl"
        cascade.serialise(str(filepath))

        loaded = Cascade.from_serialised(str(filepath))
        assert loaded is not None
