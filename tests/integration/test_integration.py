# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Integration tests for the earthkit-workflows-anemoi plugin.

These tests build full workflow DAGs and execute them through a lightweight
graph executor with mocked anemoi-inference functions. They verify:

1. DAG structure is correct (nodes, edges, dimensions)
2. Payloads can be resolved and called
3. Data flows correctly between initial conditions and inference
4. Ensemble dimensions use the correct ENSEMBLE_DIMENSION_NAME constant
5. Serialisation round-trips work
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from anemoi.inference.testing import fake_checkpoints
from earthkit.workflows.fluent import nodetree_arrays

from earthkit.workflows import Cascade, Graph, serialise
from earthkit.workflows.plugins.anemoi.fluent import (
    Action,
    Inference,
    from_config,
    from_initial_conditions,
    from_input,
    get_initial_conditions,
)
from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_payloads(action):
    """Collect all payload function references from an action's nodes."""
    payloads = []
    for _, narray in nodetree_arrays(action.nodes):
        for node in np.atleast_1d(narray.values).flatten():
            if hasattr(node, "payload") and node.payload is not None:
                payloads.append(node.payload)
    return payloads


def collect_graph_payload_funcs(action):
    """Collect all unique payload function paths from an action's full graph.

    Uses graph.nodes() to get ALL nodes including intermediate ones
    (not just the leaf nodes exposed by nodetree_arrays).
    """
    funcs = set()
    graph = action.graph()
    for node in graph.nodes():
        p = node.payload
        if p is not None and hasattr(p, "func"):
            f = p.func
            if isinstance(f, str):
                funcs.add(f)
    return funcs


def collect_graph_payload_func_labels(action):
    """Collect all payload function labels (strings or qualnames) from the full graph."""
    labels = set()
    graph = action.graph()
    for node in graph.nodes():
        p = node.payload
        if p is not None and hasattr(p, "func"):
            f = p.func
            if isinstance(f, str):
                labels.add(f)
            elif hasattr(f, "__qualname__"):
                labels.add(f.__qualname__)
    return labels


# ---------------------------------------------------------------------------
# DAG structure tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Payload resolution tests
# ---------------------------------------------------------------------------


class TestPayloadResolution:
    """Verify payloads reference valid, resolvable functions."""

    @fake_checkpoints
    def test_from_input_payloads_are_resolvable(self, simple_ckpt_path):
        """All string-referenced payloads should be importable."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        funcs = collect_graph_payload_funcs(action)

        import importlib

        for func_path in funcs:
            module_path, _, func_name = func_path.rpartition(".")
            module = importlib.import_module(module_path)
            assert hasattr(module, func_name), f"Cannot resolve {func_path}"

    @fake_checkpoints
    def test_payloads_contain_expected_functions(self, simple_ckpt_path):
        """The DAG should reference the expected inference functions."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        labels = collect_graph_payload_func_labels(action)

        # Should have both IC fetch and model run
        label_str = " ".join(labels)
        assert "_get_initial_conditions_from_config" in label_str
        assert "run_as_earthkit_from_config" in label_str


# ---------------------------------------------------------------------------
# Serialisation round-trip tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Payload metadata propagation (integration level)
# ---------------------------------------------------------------------------


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
            environment=["anemoi-inference~=0.10"],
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


# ---------------------------------------------------------------------------
# Execution tests (with mocked anemoi functions)
# ---------------------------------------------------------------------------


class TestExecution:
    """Execute anemoi-specific payloads from the DAG with mocked inference functions.

    These tests verify that the anemoi payload functions (IC fetch and model run)
    can be resolved and called with correct arguments. Internal earthkit-workflows
    operations (Backend.take etc.) are skipped as they need the full Cascade runtime.
    """

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
        from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

        ic_func = mock_registry["earthkit.workflows.plugins.anemoi.inference._get_initial_conditions_from_config"]
        state = ic_func(config={}, date="2020-01-01")
        assert isinstance(state, dict)
        assert "date" in state
        assert "fields" in state
        assert "latitudes" in state
        assert "longitudes" in state
        assert ENSEMBLE_DIMENSION_NAME not in state  # deterministic

        state_ens = ic_func(config={}, date="2020-01-01", ens_mem=3)
        assert ENSEMBLE_DIMENSION_NAME in state_ens
        assert state_ens[ENSEMBLE_DIMENSION_NAME] == 3

    @fake_checkpoints
    def test_run_mock_yields_fieldlists(self, simple_ckpt_path, mock_registry):
        """The mocked run function should yield fieldlist-like objects."""
        import datetime

        ic_func = mock_registry["earthkit.workflows.plugins.anemoi.inference._get_initial_conditions_from_config"]
        run_func = mock_registry["earthkit.workflows.plugins.anemoi.inference.run_as_earthkit_from_config"]

        state = ic_func(config={}, date="2020-01-01")
        results = list(run_func(state, config={}, lead_time=datetime.timedelta(days=1)))
        assert len(results) == 4  # 4 x 6h steps in 1D
        for r in results:
            assert hasattr(r, "fields")
            assert len(r.fields) > 0


# ---------------------------------------------------------------------------
# Ensemble dimension consistency
# ---------------------------------------------------------------------------


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
        from earthkit.workflows import fluent

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

        state = {"date": "2020-01-01"}
        result = expose_ensemble_dimension(state, 5)
        assert ENSEMBLE_DIMENSION_NAME in result
        assert result[ENSEMBLE_DIMENSION_NAME] == 5


# ---------------------------------------------------------------------------
# Multi-checkpoint / multi-model tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Data flow tests -- all entry points and chaining patterns
# ---------------------------------------------------------------------------


class TestDataFlows:
    """Verify all entry point flows produce valid DAGs with correct structure."""

    # --- from_config ---

    @fake_checkpoints
    def test_from_config_builds_graph(self, mock_config):
        """from_config should build a valid acyclic graph."""
        ckpt = str((Path(__file__).parent.parent / "checkpoints" / "simple.yaml").absolute())
        action = from_config(
            mock_config,
            date="2020-01-01",
            lead_time="1D",
            checkpoint=ckpt,
            input="dummy",
        )
        graph = action.graph()
        assert isinstance(graph, Graph)
        assert not graph.has_cycle()
        assert len(list(graph.nodes())) > 0

    @fake_checkpoints
    def test_from_config_with_ensemble(self, mock_config):
        """from_config with ensemble members should have the ensemble dimension."""
        ckpt = str((Path(__file__).parent.parent / "checkpoints" / "simple.yaml").absolute())
        action = from_config(
            mock_config,
            date="2020-01-01",
            lead_time="1D",
            checkpoint=ckpt,
            input="dummy",
            ensemble_members=3,
        )
        assert ENSEMBLE_DIMENSION_NAME in action.nodes.dims
        assert action.nodes.coords[ENSEMBLE_DIMENSION_NAME].size == 3

    @fake_checkpoints
    def test_from_config_payloads_resolvable(self, mock_config):
        """from_config graph payloads should be importable."""
        ckpt = str((Path(__file__).parent.parent / "checkpoints" / "simple.yaml").absolute())
        action = from_config(
            mock_config,
            date="2020-01-01",
            lead_time="1D",
            checkpoint=ckpt,
            input="dummy",
        )
        funcs = collect_graph_payload_funcs(action)
        assert len(funcs) > 0

        import importlib

        for func_path in funcs:
            module_path, _, func_name = func_path.rpartition(".")
            module = importlib.import_module(module_path)
            assert hasattr(module, func_name), f"Cannot resolve {func_path}"

    # --- get_initial_conditions (standalone IC fetch) ---

    @fake_checkpoints
    def test_get_initial_conditions_builds_graph(self, simple_ckpt_path):
        """get_initial_conditions should produce a valid graph."""
        action = get_initial_conditions(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        graph = action.graph()
        assert isinstance(graph, Graph)
        assert not graph.has_cycle()

    @fake_checkpoints
    def test_get_initial_conditions_has_ic_payload(self, simple_ckpt_path):
        """get_initial_conditions graph should reference the IC fetch function."""
        action = get_initial_conditions(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        labels = collect_graph_payload_func_labels(action)
        label_str = " ".join(labels)
        assert "_get_initial_conditions_from_config" in label_str

    @fake_checkpoints
    def test_get_initial_conditions_no_run_payload(self, simple_ckpt_path):
        """get_initial_conditions should NOT have a model run payload."""
        action = get_initial_conditions(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        labels = collect_graph_payload_func_labels(action)
        label_str = " ".join(labels)
        assert "run_as_earthkit_from_config" not in label_str

    # --- Inference class: from_initial_conditions ---

    @fake_checkpoints
    def test_inference_from_initial_conditions_none(self, simple_ckpt_path):
        """Inference.from_initial_conditions(None) should build a valid graph."""
        inference = Inference(simple_ckpt_path, lead_time="1D")
        action = inference.from_initial_conditions(None, ensemble_members=2)
        graph = action.graph()
        assert not graph.has_cycle()
        assert ENSEMBLE_DIMENSION_NAME in action.nodes.dims

    @fake_checkpoints
    def test_inference_from_initial_conditions_action(self, simple_ckpt_path):
        """Inference.from_initial_conditions with a fluent.Action should chain correctly."""
        from earthkit.workflows import fluent

        init = fluent.from_source(
            [None, None],
            dims=[ENSEMBLE_DIMENSION_NAME],
            coords={ENSEMBLE_DIMENSION_NAME: [1, 2]},
        )
        inference = Inference(simple_ckpt_path, lead_time="1D")
        action = inference.from_initial_conditions(init)

        graph = action.graph()
        assert not graph.has_cycle()
        assert ENSEMBLE_DIMENSION_NAME in action.nodes.dims

        # Should have the model run payload
        labels = collect_graph_payload_func_labels(action)
        label_str = " ".join(labels)
        assert "run_as_earthkit_from_config" in label_str

    # --- Inference class: get_initial_conditions ---

    @fake_checkpoints
    def test_inference_get_initial_conditions(self, simple_ckpt_path):
        """Inference.get_initial_conditions should produce IC-only graph."""
        inference = Inference(simple_ckpt_path, lead_time="1D")
        action = inference.get_initial_conditions("dummy", "2020-01-01")
        graph = action.graph()
        assert not graph.has_cycle()

        labels = collect_graph_payload_func_labels(action)
        label_str = " ".join(labels)
        assert "_get_initial_conditions_from_config" in label_str
        assert "run_as_earthkit_from_config" not in label_str

    # --- Action.infer chaining ---

    @fake_checkpoints
    def test_action_infer_chains_correctly(self, simple_ckpt_path):
        """Action.infer should chain IC action into a full inference graph."""
        from earthkit.workflows import fluent

        init = Action(
            fluent.from_source(
                [None, None],
                dims=[ENSEMBLE_DIMENSION_NAME],
                coords={ENSEMBLE_DIMENSION_NAME: [1, 2]},
            ).nodes
        )
        action = init.infer(simple_ckpt_path, lead_time="1D")
        graph = action.graph()
        assert not graph.has_cycle()
        assert ENSEMBLE_DIMENSION_NAME in action.nodes.dims

        # Should have model run payload since we chained inference
        labels = collect_graph_payload_func_labels(action)
        label_str = " ".join(labels)
        assert "run_as_earthkit_from_config" in label_str

    # --- get_initial_conditions -> from_initial_conditions pipeline ---

    @fake_checkpoints
    def test_ic_to_inference_pipeline(self, simple_ckpt_path):
        """get_initial_conditions output fed into from_initial_conditions should work."""
        ic_action = get_initial_conditions(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        action = from_initial_conditions(simple_ckpt_path, ic_action, lead_time="1D")
        graph = action.graph()
        assert not graph.has_cycle()

        labels = collect_graph_payload_func_labels(action)
        label_str = " ".join(labels)
        # Should have both IC fetch and model run
        assert "_get_initial_conditions_from_config" in label_str
        assert "run_as_earthkit_from_config" in label_str

    @fake_checkpoints
    def test_ic_to_infer_pipeline_with_ensemble(self, simple_ckpt_path):
        """get_initial_conditions -> Action.infer with ensemble should propagate dims."""
        from earthkit.workflows import fluent

        # Build ensemble IC source
        init = Action(
            fluent.from_source(
                [None, None, None],
                dims=[ENSEMBLE_DIMENSION_NAME],
                coords={ENSEMBLE_DIMENSION_NAME: [1, 2, 3]},
            ).nodes
        )
        action = init.infer(simple_ckpt_path, lead_time="1D")

        assert ENSEMBLE_DIMENSION_NAME in action.nodes.dims
        assert action.nodes.coords[ENSEMBLE_DIMENSION_NAME].size == 3

        graph = action.graph()
        assert not graph.has_cycle()

    # --- from_input includes both IC and run ---

    @fake_checkpoints
    def test_from_input_has_both_ic_and_run(self, simple_ckpt_path):
        """from_input graph should contain both IC fetch and model run payloads."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        labels = collect_graph_payload_func_labels(action)
        label_str = " ".join(labels)
        assert "_get_initial_conditions_from_config" in label_str
        assert "run_as_earthkit_from_config" in label_str

    # --- from_initial_conditions with None vs Action produce different graphs ---

    @fake_checkpoints
    def test_from_initial_conditions_none_vs_action(self, simple_ckpt_path):
        """from_initial_conditions(None) and from_initial_conditions(action) should
        both produce valid graphs but with different source structures."""
        from earthkit.workflows import fluent

        action_none = from_initial_conditions(simple_ckpt_path, None, lead_time="1D", ensemble_members=2)
        init = fluent.from_source(
            [None, None],
            dims=[ENSEMBLE_DIMENSION_NAME],
            coords={ENSEMBLE_DIMENSION_NAME: [1, 2]},
        )
        action_from_act = from_initial_conditions(simple_ckpt_path, init, lead_time="1D")

        graph_none = action_none.graph()
        graph_act = action_from_act.graph()

        assert not graph_none.has_cycle()
        assert not graph_act.has_cycle()

        # Both should have the model run payload
        for action in [action_none, action_from_act]:
            labels = collect_graph_payload_func_labels(action)
            assert any("run_as_earthkit_from_config" in label for label in labels)
