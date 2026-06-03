# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests verifying metadata flows through the full pipeline."""

from __future__ import annotations

import pytest
from anemoi.inference.testing import fake_checkpoints

from earthkit.workflows import serialise
from earthkit.workflows.plugins.anemoi.fluent import from_input

pytestmark = pytest.mark.integration


class TestPayloadMetadataIntegration:
    """Verify metadata flows through the full pipeline."""

    @fake_checkpoints
    def test_environment_metadata_on_anemoi_nodes(self, simple_ckpt_path):
        """Anemoi payload nodes (string-referenced) should carry environment metadata."""
        action = from_input(
            simple_ckpt_path,
            "dummy",
            date="2020-01-01",
            lead_time="1D",
            environment=["anemoi-inference~=0.11"],
        )
        graph = action.graph()
        anemoi_nodes_found = 0
        for node in graph.nodes():
            p = node.payload
            if p is not None and hasattr(p, "func") and isinstance(p.func, str):
                anemoi_nodes_found += 1
                assert (
                    "environment" in p.metadata
                ), f"Node {node.name!r} with func {p.func!r} missing environment metadata"
        assert anemoi_nodes_found > 0, "No anemoi payload nodes found in graph"

    @fake_checkpoints
    def test_custom_payload_metadata_survives_graph(self, simple_ckpt_path):
        """Custom payload_metadata should be on every node after graph construction."""
        custom = {"experiment_id": "test-001", "run_type": "integration"}
        action = from_input(
            simple_ckpt_path,
            "dummy",
            date="2020-01-01",
            lead_time="1D",
            payload_metadata=custom,
        )
        graph = action.graph()
        serialised = serialise(graph)

        # The metadata should survive serialisation
        for node_name, node_data in serialised.items():
            if "payload" in node_data:
                payload = node_data["payload"]
                if isinstance(payload, dict) and "metadata" in payload:
                    for key in custom:
                        assert key in payload["metadata"]

    @fake_checkpoints
    def test_run_as_earthkit_has_needs_gpu(self, simple_ckpt_path):
        """The run_as_earthkit payload should have needs_gpu=True in metadata."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        graph = action.graph()
        model_nodes_found = 0
        for node in graph.nodes():
            p = node.payload
            if p is not None and hasattr(p, "func") and isinstance(p.func, str) and "run_as_earthkit" in p.func:
                model_nodes_found += 1
                assert p.metadata.get("needs_gpu") is True, (
                    f"Node {node.name!r} with func {p.func!r} should have needs_gpu=True "
                    f"in metadata, got {p.metadata}"
                )
        assert model_nodes_found > 0, "No run_as_earthkit nodes found in graph"
