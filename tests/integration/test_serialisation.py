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

from earthkit.workflows.graph import serialise
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