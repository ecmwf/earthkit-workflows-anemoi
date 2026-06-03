# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests verifying payloads reference valid, resolvable functions."""

from __future__ import annotations

import importlib

import pytest
from anemoi.inference.testing import fake_checkpoints

from earthkit.workflows.plugins.anemoi.fluent import from_input

from .conftest import collect_graph_payload_func_labels, collect_graph_payload_funcs

pytestmark = pytest.mark.integration


class TestPayloadResolution:
    """Verify payloads reference valid, resolvable functions."""

    @fake_checkpoints
    def test_from_input_payloads_are_resolvable(self, simple_ckpt_path):
        """All string-referenced payloads should be importable."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        funcs = collect_graph_payload_funcs(action)

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
        assert "_get_initial_conditions" in label_str
        assert "run_as_earthkit" in label_str
