# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Fixtures for integration tests.

Provides a lightweight graph executor that traverses the DAG built by the
fluent API and calls each payload's function, with anemoi-inference
functions mocked so that no GPU or model weights are required.
"""

from __future__ import annotations

import datetime
import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing.mock_checkpoint import MockRunConfiguration
from earthkit.workflows.fluent import Action, Payload, nodetree_arrays
from earthkit.workflows.graph import Node, Output

from earthkit.workflows import serialise

# ---------------------------------------------------------------------------
# Shared test helpers (used across multiple test modules)
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
# Helpers: resolve and execute payloads
# ---------------------------------------------------------------------------


def _resolve_func(func):
    """Resolve a string function reference to a callable."""
    if isinstance(func, str):
        module_path, _, func_name = func.rpartition(".")
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    return func


def _make_fake_fieldlist(step: int, ensemble_member: int | None = None):
    """Create a minimal fake fieldlist-like object for test outputs."""
    fields = []
    for param in ["2t", "10u", "10v", "msl", "tcc", "tp"]:
        field = MagicMock()
        field.metadata.return_value = {
            "param": param,
            "step": step,
            "number": ensemble_member,
        }
        field.values = np.random.randn(100)
        fields.append(field)

    fieldlist = MagicMock()
    fieldlist.fields = fields
    fieldlist.__iter__ = lambda self: iter(fields)
    fieldlist.__len__ = lambda self: len(fields)
    return fieldlist


def _make_fake_state(date: datetime.datetime, step_hours: int = 6) -> dict:
    """Create a minimal fake anemoi state dict."""
    return {
        "date": date + datetime.timedelta(hours=step_hours),
        "latitudes": np.linspace(-90, 90, 100),
        "longitudes": np.linspace(0, 360, 100),
        "fields": {
            "2t": np.random.randn(100),
            "10u": np.random.randn(100),
            "10v": np.random.randn(100),
            "msl": np.random.randn(100),
            "tcc": np.random.randn(100),
            "tp": np.random.randn(100),
        },
    }


# ---------------------------------------------------------------------------
# Mock implementations of anemoi inference functions
# ---------------------------------------------------------------------------


def mock_get_initial_conditions(config, date, number=None, **kwargs):
    """Mock replacement for _get_initial_conditions.

    Returns dict[str, State] to match the real multi-dataset API.
    """
    from earthkit.data.utils.dates import to_datetime

    from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

    state = {
        "date": to_datetime(date),
        "latitudes": np.linspace(-90, 90, 100),
        "longitudes": np.linspace(0, 360, 100),
        "fields": {
            "2t": np.random.randn(100),
            "10u": np.random.randn(100),
            "10v": np.random.randn(100),
            "msl": np.random.randn(100),
            "tcc": np.random.randn(100),
            "tp": np.random.randn(100),
        },
    }
    if number is not None:
        state[ENSEMBLE_DIMENSION_NAME] = number

    # Return dict[str, State] for multi-dataset support
    return {"era5": state}


def mock_run_as_earthkit(input_state, config, lead_time, **kwargs):
    """Mock replacement for run_as_earthkit.

    Yields dict[str, SimpleFieldList] per step to match the real multi-dataset API.
    """
    from anemoi.utils.dates import frequency_to_seconds

    from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

    lead_time_seconds = frequency_to_seconds(lead_time)
    model_step = 6 * 3600  # 6h steps

    # Handle both single state and dict-of-states input
    if isinstance(input_state, dict) and "fields" in input_state:
        # Old-style single state (backward compat)
        ensemble_member = input_state.get(ENSEMBLE_DIMENSION_NAME, None)
    elif isinstance(input_state, dict):
        # New dict-of-datasets style - get ensemble from first dataset
        ensemble_member = next(iter(input_state.values())).get(ENSEMBLE_DIMENSION_NAME, None)
    else:
        ensemble_member = None

    for step_seconds in range(model_step, lead_time_seconds + model_step, model_step):
        step_hours = step_seconds // 3600
        # Yield dict[str, SimpleFieldList] for multi-dataset support
        yield {"era5": _make_fake_fieldlist(step_hours, ensemble_member)}


# ---------------------------------------------------------------------------
# Graph executor
# ---------------------------------------------------------------------------


class SimpleGraphExecutor:
    """Execute a fluent Action's graph by topologically traversing nodes.

    This is a minimal executor for integration testing. It resolves payload
    functions, calls them with their arguments, and collects results.
    Payloads whose functions are in the mock registry get replaced with mocks.
    """

    def __init__(self, mock_registry: dict[str, Any] | None = None):
        self.mock_registry = mock_registry or {}
        self.results: dict[str, Any] = {}

    def execute(self, action: Action) -> dict[str, Any]:
        """Execute all nodes in the action's graph."""
        graph = action.graph()
        serialised = serialise(graph)

        # Topological order: sources first
        ordered = list(graph.nodes(forwards=True))

        for node in ordered:
            self._execute_node(node, serialised)

        return self.results

    def _execute_node(self, node: Node, serialised: dict) -> Any:
        if node.name in self.results:
            return self.results[node.name]

        # Resolve inputs
        input_values = {}
        for input_name, output in node.inputs.items():
            parent = output.parent if isinstance(output, Output) else output
            if parent.name not in self.results:
                self._execute_node(parent, serialised)
            input_values[input_name] = self.results[parent.name]

        # Execute payload
        payload = node.payload
        if payload is None:
            self.results[node.name] = None
            return None

        if isinstance(payload, Payload):
            func = payload.func
            func_key = func if isinstance(func, str) else getattr(func, "__qualname__", str(func))

            # Check mock registry
            if func_key in self.mock_registry:
                func = self.mock_registry[func_key]
            else:
                func = _resolve_func(func)

            # Build args, replacing Node.input_name references with actual values

            args = []
            for arg in payload.args:
                if isinstance(arg, str) and arg.startswith("input"):
                    # This is a reference to an input node
                    matching = [v for k, v in input_values.items()]
                    if matching:
                        args.append(matching[0])
                    else:
                        args.append(arg)
                else:
                    args.append(arg)

            kwargs = dict(payload.kwargs)

            try:
                result = func(*args, **kwargs)
                # If it's a generator, consume it
                if hasattr(result, "__next__"):
                    result = list(result)
            except Exception as e:
                result = f"ERROR: {e}"

            self.results[node.name] = result
            return result
        else:
            # Raw callable payload
            if callable(payload):
                result = payload()
            else:
                result = payload
            self.results[node.name] = result
            return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_registry():
    """Registry mapping function paths to their mock replacements."""
    return {
        "earthkit.workflows.plugins.anemoi.inference._get_initial_conditions": mock_get_initial_conditions,
        "earthkit.workflows.plugins.anemoi.inference.run_as_earthkit": mock_run_as_earthkit,
    }


@pytest.fixture
def executor(mock_registry):
    """A SimpleGraphExecutor wired up with mocks."""
    return SimpleGraphExecutor(mock_registry)


@pytest.fixture
@fake_checkpoints
def simple_ckpt_path():
    return str((Path(__file__).parent.parent / "checkpoints" / "simple.yaml").absolute())


@pytest.fixture
@fake_checkpoints
def full_atmo_ckpt_path():
    return str((Path(__file__).parent.parent / "checkpoints" / "full_atmo.yaml").absolute())


@pytest.fixture
@fake_checkpoints
def mock_config(tmp_path: Path):
    parent_dir = Path(__file__).parent.parent
    config_path = parent_dir / "configs" / "simple.yaml"

    config_dict = yaml.safe_load(config_path.read_text())
    config_dict["checkpoint"] = f"{parent_dir}/{config_dict['checkpoint']}"

    with open(tmp_path / "simple.yaml", "w") as f:
        yaml.safe_dump(config_dict, f)

    tmp_path = tmp_path / "simple.yaml"

    return MockRunConfiguration.load(
        str(tmp_path.absolute()),
        overrides=dict(runner="testing", device="cpu", input="dummy"),
    )
