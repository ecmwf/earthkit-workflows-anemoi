# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for dict-of-qubes expansion and dataset dimension propagation.

These tests verify that:
1. expansion_qube_from_metadata / expansion_qube_from_variables return dict[str, Qube]
2. The dataset dimension is added correctly by _run_model
3. Single-Qube wrapping via Inference selects the dataset back out
4. Multi-dataset qubes produce the correct dataset dimension values
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
from anemoi.inference.testing import fake_checkpoints
from qubed import Qube
from xarray import DataArray, DataTree

from earthkit.workflows.plugins.anemoi.fluent import (
    DEFAULT_DATASET_NAME,
    Inference,
    _default_dictionarify,
    from_initial_conditions,
    from_input,
)
from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME
from earthkit.workflows.plugins.anemoi.utils import expansion_qube_from_metadata, expansion_qube_from_variables

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_dims_and_coords(action):
    """Collect all dims and coord keys from an action's node tree."""
    all_dims = set()
    all_coords = set()
    all_coord_values = defaultdict(set)

    def walk(node: DataTree | DataArray):
        if isinstance(node, DataTree):
            all_dims.update(node.dims)
            all_coords.update(node.coords.keys())
            for dim in node.coords:
                all_coord_values[dim].update(np.atleast_1d(node.coords[dim].values))
            for child in node.children:
                walk(node[child])
        elif isinstance(node, DataArray):
            all_dims.update(node.dims)
            all_coords.update(node.coords.keys())
            for dim in node.coords:
                all_coord_values[dim].update(np.atleast_1d(node.coords[dim].values))

    walk(action.nodes)
    return all_dims, all_coords, all_coord_values


# ---------------------------------------------------------------------------
# expansion_qube_from_metadata returns dict[str, Qube]
# ---------------------------------------------------------------------------


class TestExpansionQubeReturnType:
    """Verify expansion qube functions return dict[str, Qube]."""

    @fake_checkpoints
    def test_from_metadata_returns_dict(self, simple_ckpt_path):
        """expansion_qube_from_metadata should return a dict of Qubes."""
        from anemoi.inference.checkpoint import Checkpoint

        metadata = Checkpoint(simple_ckpt_path).multi_dataset_metadata
        qubes = expansion_qube_from_metadata(metadata, "1D")

        assert isinstance(qubes, dict)
        assert len(qubes) > 0
        for key, qube in qubes.items():
            assert isinstance(key, str), f"Key should be str, got {type(key)}"
            assert isinstance(qube, Qube), f"Value for {key!r} should be Qube, got {type(qube)}"

    @fake_checkpoints
    def test_from_metadata_qubes_have_step_and_param(self, simple_ckpt_path):
        """Each Qube in the dict should have step and param axes."""
        from anemoi.inference.checkpoint import Checkpoint

        metadata = Checkpoint(simple_ckpt_path).multi_dataset_metadata
        qubes = expansion_qube_from_metadata(metadata, "1D")

        for key, qube in qubes.items():
            axes = qube.axes()
            assert "step" in axes, f"Qube {key!r} missing 'step' axis"
            assert "param" in axes, f"Qube {key!r} missing 'param' axis"

    @fake_checkpoints
    def test_from_variables_returns_dict(self, simple_ckpt_path):
        """expansion_qube_from_variables should return a dict of Qubes."""
        from anemoi.inference.checkpoint import Checkpoint

        ckpt = Checkpoint(simple_ckpt_path)
        md = ckpt.multi_dataset_metadata
        # Build the variables dict matching the metadata dict structure
        variables = {
            ds: m.select_variables(include=["diagnostic", "prognostic"], has_mars_requests=False)
            for ds, m in md.items()
        }

        variables_metadata = next(iter(md.values())).typed_variables
        model_step = next(iter(md.values())).timestep.seconds

        qubes = expansion_qube_from_variables(variables, variables_metadata, model_step, "1D")

        assert isinstance(qubes, dict)
        assert len(qubes) > 0
        for key, qube in qubes.items():
            assert isinstance(key, str)
            assert isinstance(qube, Qube)

    @fake_checkpoints
    def test_from_metadata_and_variables_match(self, simple_ckpt_path):
        """Both functions should produce equivalent qubes for the same checkpoint."""
        from anemoi.inference.checkpoint import Checkpoint

        ckpt = Checkpoint(simple_ckpt_path)
        md = ckpt.multi_dataset_metadata

        qubes_from_meta = expansion_qube_from_metadata(md, "1D")

        variables = {
            ds: m.select_variables(include=["diagnostic", "prognostic"], has_mars_requests=False)
            for ds, m in md.items()
        }
        variables_metadata = next(iter(md.values())).typed_variables
        model_step = next(iter(md.values())).timestep.seconds
        qubes_from_vars = expansion_qube_from_variables(variables, variables_metadata, model_step, "1D")

        assert set(qubes_from_meta.keys()) == set(qubes_from_vars.keys())
        for key in qubes_from_meta:
            assert qubes_from_meta[key].axes() == qubes_from_vars[key].axes()

    @fake_checkpoints
    def test_full_atmo_has_level_axis(self, full_atmo_ckpt_path):
        """full_atmo checkpoint qubes should contain level axes."""
        from anemoi.inference.checkpoint import Checkpoint

        metadata = Checkpoint(full_atmo_ckpt_path).multi_dataset_metadata
        qubes = expansion_qube_from_metadata(metadata, "1D")

        # At least one dataset's qube should have pressure levels
        has_level = any("level" in qube.axes() for qube in qubes.values())
        assert has_level, "full_atmo qubes should have at least one dataset with a 'level' axis"


# ---------------------------------------------------------------------------
# _default_dictionarify wraps single Qube correctly
# ---------------------------------------------------------------------------


class TestDefaultDictionarify:
    """Verify _default_dictionarify wraps single Qubes into a dict."""

    def test_single_qube_wrapped(self):
        """A single Qube should be wrapped with DEFAULT_DATASET_NAME key."""
        qube = Qube.from_datacube({"step": [6, 12], "param": ["2t", "msl"]})
        result = _default_dictionarify(qube, Qube)

        assert isinstance(result, dict)
        assert DEFAULT_DATASET_NAME in result
        assert result[DEFAULT_DATASET_NAME] is qube

    def test_dict_of_qubes_passed_through(self):
        """A dict[str, Qube] should pass through unchanged."""
        q1 = Qube.from_datacube({"step": [6, 12], "param": ["2t"]})
        q2 = Qube.from_datacube({"step": [6, 12], "param": ["q"]})
        d = {"surface": q1, "pressure": q2}
        result = _default_dictionarify(d, Qube)

        assert result is d

    def test_plain_dict_wrapped(self):
        """A plain dict (not containing Qubes) should be wrapped with DEFAULT_DATASET_NAME."""
        d = {"step": [6, 12]}
        result = _default_dictionarify(d, Qube)

        assert isinstance(result, dict)
        assert DEFAULT_DATASET_NAME in result


# ---------------------------------------------------------------------------
# Dataset dimension in the action tree
# ---------------------------------------------------------------------------


class TestDatasetDimension:
    """Verify the dataset dimension is added correctly by _run_model."""

    @fake_checkpoints
    def test_from_input_has_dataset_dim(self, simple_ckpt_path):
        """from_input with default (multi-dataset) expansion should have a dataset dimension."""
        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        _, all_coords, _ = collect_dims_and_coords(action)

        assert "dataset" in all_coords, f"Expected 'dataset' in coords, got: {all_coords}"

    @fake_checkpoints
    def test_from_input_dataset_values_match_metadata_keys(self, simple_ckpt_path):
        """Dataset dimension values should match the keys from the expansion qube dict."""
        from anemoi.inference.checkpoint import Checkpoint
        from anemoi.utils.dates import as_timedelta

        metadata = Checkpoint(simple_ckpt_path).multi_dataset_metadata
        expected_keys = set(expansion_qube_from_metadata(metadata, as_timedelta("1D")).keys())

        action = from_input(simple_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        _, _, coord_values = collect_dims_and_coords(action)

        assert "dataset" in coord_values
        actual_datasets = coord_values["dataset"]
        assert actual_datasets == expected_keys, f"Expected dataset values {expected_keys}, got {actual_datasets}"

    @fake_checkpoints
    def test_from_initial_conditions_has_dataset_dim(self, simple_ckpt_path):
        """from_initial_conditions should also propagate the dataset dimension."""
        action = from_initial_conditions(simple_ckpt_path, None, lead_time="1D")
        _, all_coords, _ = collect_dims_and_coords(action)

        assert "dataset" in all_coords

    @fake_checkpoints
    def test_explicit_multi_dataset_qube_produces_correct_dims(self, simple_ckpt_path):
        """Passing a dict[str, Qube] should produce dataset dimension with those keys."""
        q_sfc = Qube.from_datacube({"step": [6, 12, 18, 24], "levtype": ["sfc"], "param": ["2t", "msl"]})
        q_pl = Qube.from_datacube({"step": [6, 12, 18, 24], "levtype": ["pl"], "param": ["q"], "level": [850]})
        multi_qube = {"surface": q_sfc, "pressure": q_pl}

        action = from_initial_conditions(simple_ckpt_path, None, lead_time="1D", expansion_qube=multi_qube)
        _, _, coord_values = collect_dims_and_coords(action)

        assert "dataset" in coord_values
        assert coord_values["dataset"] == {"surface", "pressure"}

    @fake_checkpoints
    def test_single_qube_no_dataset_dim(self, simple_ckpt_path):
        """When a single Qube is passed, the dataset dim should be selected away."""
        qube = Qube.from_datacube({"step": [6, 12, 18, 24], "levtype": ["sfc"], "param": ["2t", "msl"]})

        action = from_initial_conditions(simple_ckpt_path, None, lead_time="1D", expansion_qube=qube)
        all_dims, _, _ = collect_dims_and_coords(action)

        # Single qube: dataset dimension should not appear as a dimension
        assert "dataset" not in all_dims, "Single Qube should have dataset selected out, not appear as a dimension"


# ---------------------------------------------------------------------------
# Inference class wrapping behaviour
# ---------------------------------------------------------------------------


class TestInferenceQubeWrapping:
    """Verify the Inference class handles single vs dict qubes correctly."""

    @fake_checkpoints
    def test_inference_single_qube_wraps_to_dict(self, simple_ckpt_path):
        """Inference with a single Qube should wrap it in a dict internally."""
        qube = Qube.from_datacube({"step": [6, 12, 18, 24], "param": ["2t", "msl"]})
        inference = Inference(simple_ckpt_path, lead_time="1D", expansion_qube=qube)

        assert isinstance(inference.expansion_qube, dict)
        assert DEFAULT_DATASET_NAME in inference.expansion_qube

    @fake_checkpoints
    def test_inference_dict_qube_kept_as_dict(self, simple_ckpt_path):
        """Inference with a dict[str, Qube] should keep it as-is."""
        q1 = Qube.from_datacube({"step": [6, 12], "param": ["2t"]})
        q2 = Qube.from_datacube({"step": [6, 12], "param": ["q"]})
        multi = {"surface": q1, "pressure": q2}
        inference = Inference(simple_ckpt_path, lead_time="1D", expansion_qube=multi)

        assert inference.expansion_qube is multi
        assert not inference._given_single_qube

    @fake_checkpoints
    def test_inference_from_metadata_produces_dict(self, simple_ckpt_path):
        """Inference loading from checkpoint metadata should produce a dict qube."""
        inference = Inference(simple_ckpt_path, lead_time="1D")

        assert isinstance(inference.expansion_qube, dict)
        assert len(inference.expansion_qube) > 0
        for v in inference.expansion_qube.values():
            assert isinstance(v, Qube)

    @fake_checkpoints
    def test_inference_single_qube_selects_dataset(self, simple_ckpt_path):
        """Inference with single Qube should select away the dataset dimension in results."""
        qube = Qube.from_datacube({"step": [6, 12, 18, 24], "param": ["2t", "msl"]})
        inference = Inference(simple_ckpt_path, lead_time="1D", expansion_qube=qube)
        action = inference.from_initial_conditions(None)

        all_dims, _, _ = collect_dims_and_coords(action)
        assert "dataset" not in all_dims

    @fake_checkpoints
    def test_inference_multi_qube_keeps_dataset_dim(self, simple_ckpt_path):
        """Inference with dict[str, Qube] should keep the dataset dimension."""
        q1 = Qube.from_datacube({"step": [6, 12, 18, 24], "param": ["2t"]})
        q2 = Qube.from_datacube({"step": [6, 12, 18, 24], "param": ["q"]})
        multi = {"atmo": q1, "ocean": q2}
        inference = Inference(simple_ckpt_path, lead_time="1D", expansion_qube=multi)
        action = inference.from_initial_conditions(None)

        _, all_coords, coord_values = collect_dims_and_coords(action)
        assert "dataset" in all_coords
        assert coord_values["dataset"] == {"atmo", "ocean"}


# ---------------------------------------------------------------------------
# Full atmo checkpoint multi-dataset structure
# ---------------------------------------------------------------------------


class TestFullAtmoDatasetDimension:
    """Verify dataset dimension with the full_atmo checkpoint."""

    @fake_checkpoints
    def test_full_atmo_dataset_dim_present(self, full_atmo_ckpt_path):
        """full_atmo from_input should have the dataset dimension."""
        action = from_input(full_atmo_ckpt_path, "dummy", date="2020-01-01", lead_time="1D")
        _, all_coords, _ = collect_dims_and_coords(action)

        assert "dataset" in all_coords

    @fake_checkpoints
    def test_full_atmo_with_ensemble_has_dataset_and_ensemble(self, full_atmo_ckpt_path):
        """full_atmo with ensemble should have both dataset and ensemble dims."""
        action = from_input(
            full_atmo_ckpt_path,
            "dummy",
            date="2020-01-01",
            lead_time="1D",
            ensemble_members=2,
        )
        all_dims, all_coords, _ = collect_dims_and_coords(action)

        assert ENSEMBLE_DIMENSION_NAME in all_dims
        assert "dataset" in all_coords
