# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests verifying the structure of DAGs built by the fluent API."""

from __future__ import annotations

import pytest
from anemoi.inference.testing import fake_checkpoints

from earthkit.workflows import Graph
from earthkit.workflows.plugins.anemoi.fluent import Inference, from_initial_conditions, from_input
from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

pytestmark = pytest.mark.integration


class TestDAGStructure:
    """Verify the structure of DAGs built by the fluent API."""

    @fake_checkpoints
    def test_from_input_builds_valid_graph(self, simple_ckpt_path):
        """from_input should produce a DAG that can be converted to a Graph."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        graph = action.graph()

        assert isinstance(graph, Graph)
        nodes = list(graph.nodes())
        assert len(nodes) > 0

    @fake_checkpoints
    def test_from_input_graph_is_acyclic(self, simple_ckpt_path):
        """The DAG must not contain cycles."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        graph = action.graph()
        assert not graph.has_cycle()

    @fake_checkpoints
    def test_from_input_has_source_nodes(self, simple_ckpt_path):
        """The graph should have source nodes (initial conditions)."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        graph = action.graph()
        sources = list(graph.sources())
        assert len(sources) > 0

    @fake_checkpoints
    def test_ensemble_dimension_in_graph(self, simple_ckpt_path):
        """Ensemble runs should have the correct dimension name in coords."""
        action = from_input(
            simple_ckpt_path,
            "dummy",
            date="2020-01-01",
            lead_time="1D",
            ensemble_members=3,
        )
        assert ENSEMBLE_DIMENSION_NAME in action.nodes.dims
        assert action.nodes.coords[ENSEMBLE_DIMENSION_NAME].size == 3

    @fake_checkpoints
    def test_single_member_has_ensemble_coord(self, simple_ckpt_path):
        """A deterministic run should have the ensemble name as a scalar coordinate, not a dimension."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        # Single member: ensemble name appears as a scalar coordinate (value None),
        # NOT as a dimension with size > 0
        assert ENSEMBLE_DIMENSION_NAME in action.nodes.coords

    @fake_checkpoints
    def test_from_initial_conditions_builds_graph(self, simple_ckpt_path):
        """from_initial_conditions with None should build a valid graph."""
        action = from_initial_conditions(simple_ckpt_path, None, lead_time="1D")
        graph = action.graph()
        assert isinstance(graph, Graph)
        assert not graph.has_cycle()

    @fake_checkpoints
    def test_inference_class_from_input(self, simple_ckpt_path):
        """Inference class should produce identical structure to module function."""
        inference = Inference(simple_ckpt_path, lead_time="1D")
        action = inference.from_input("dummy", "2020-01-01", ensemble_members=2)
        assert ENSEMBLE_DIMENSION_NAME in action.nodes.dims
        assert action.nodes.coords[ENSEMBLE_DIMENSION_NAME].size == 2
