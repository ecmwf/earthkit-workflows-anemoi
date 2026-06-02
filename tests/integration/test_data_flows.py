# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests verifying all entry point flows produce valid DAGs with correct structure."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from anemoi.inference.testing import fake_checkpoints

from earthkit.workflows import Graph, fluent
from earthkit.workflows.plugins.anemoi.fluent import (
    Action,
    Inference,
    from_config,
    from_initial_conditions,
    from_input,
    get_initial_conditions,
)
from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

from .conftest import collect_graph_payload_func_labels, collect_graph_payload_funcs

pytestmark = pytest.mark.integration


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
        assert "_get_initial_conditions" in label_str

    @fake_checkpoints
    def test_get_initial_conditions_no_run_payload(self, simple_ckpt_path):
        """get_initial_conditions should NOT have a model run payload."""
        action = get_initial_conditions(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        labels = collect_graph_payload_func_labels(action)
        label_str = " ".join(labels)
        assert "run_as_earthkit" not in label_str

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
        assert "run_as_earthkit" in label_str

    # --- Action.infer chaining ---

    @fake_checkpoints
    def test_action_infer_chains_correctly(self, simple_ckpt_path):
        """Action.infer should chain IC action into a full inference graph."""
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
        assert "run_as_earthkit" in label_str

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
        assert "_get_initial_conditions" in label_str
        assert "run_as_earthkit" in label_str

    @fake_checkpoints
    def test_ic_to_infer_pipeline_with_ensemble(self, simple_ckpt_path):
        """get_initial_conditions -> Action.infer with ensemble should propagate dims."""
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
        assert "_get_initial_conditions" in label_str
        assert "run_as_earthkit" in label_str

    # --- from_initial_conditions with None vs Action produce different graphs ---

    @fake_checkpoints
    def test_from_initial_conditions_none_vs_action(self, simple_ckpt_path):
        """from_initial_conditions(None) and from_initial_conditions(action) should
        both produce valid graphs but with different source structures."""
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
            assert any("run_as_earthkit" in label for label in labels)
