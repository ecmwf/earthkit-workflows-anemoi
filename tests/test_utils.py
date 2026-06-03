# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for earthkit.workflows.plugins.anemoi.utils."""

from pathlib import Path

import pytest
from anemoi.inference.testing import fake_checkpoints

from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME
from earthkit.workflows.plugins.anemoi.utils import (
    _add_self_to_environment,
    crack_environment,
    expansion_qube_from_metadata,
    expansion_qube_from_variables,
    expose_ensemble_dimension,
    faked_ensemble_transform,
    parse_ensemble_members,
)

# --- parse_ensemble_members ---


class TestParseEnsembleMembers:
    """Tests for parse_ensemble_members."""

    def test_none_returns_list_of_none(self):
        assert parse_ensemble_members(None) == [None]

    def test_single_int(self):
        result = parse_ensemble_members(3)
        assert result == [1, 2, 3]

    def test_single_int_one(self):
        result = parse_ensemble_members(1)
        assert result == [1]

    def test_sequence_passed_through(self):
        result = parse_ensemble_members([5, 10, 15])
        assert result == [5, 10, 15]

    def test_range_passed_through(self):
        result = parse_ensemble_members(range(1, 4))
        assert result == [1, 2, 3]

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            parse_ensemble_members(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            parse_ensemble_members(-1)


# --- expose_ensemble_dimension ---


class TestExposeEnsembleDimension:
    """Tests for expose_ensemble_dimension."""

    def test_adds_ensemble_key(self):
        state = {"data": {"date": "2020-01-01"}}
        result = expose_ensemble_dimension(state, 5)
        assert result["data"][ENSEMBLE_DIMENSION_NAME] == 5

    def test_none_member_no_key(self):
        state = {"data": {"date": "2020-01-01"}}
        result = expose_ensemble_dimension(state, None)
        assert ENSEMBLE_DIMENSION_NAME not in result["data"]

    def test_mutates_input_dict(self):
        """The function should mutate the dict in place and return it."""
        state = {"data": {"date": "2020-01-01"}}
        result = expose_ensemble_dimension(state, 1)
        assert result is state

    def test_non_dict_raises(self):
        with pytest.raises(AssertionError, match="dictionary"):
            expose_ensemble_dimension("not a dict", 1)

    def test_uses_constant_not_hardcoded(self):
        """Ensure the function uses the ENSEMBLE_DIMENSION_NAME constant."""
        state = {"data": {}}
        expose_ensemble_dimension(state, 42)
        assert ENSEMBLE_DIMENSION_NAME in state["data"]
        # If the constant were still 'ensemble_member', this would trivially pass.
        # The important test is that only ENSEMBLE_DIMENSION_NAME is set.
        assert len([k for k in state["data"] if k in ("ensemble_member", "number")]) == 1


# --- faked_ensemble_transform ---


class TestFakedEnsembleTransform:
    """Tests for faked_ensemble_transform."""

    def test_returns_action(self):
        from earthkit.workflows import fluent

        source = fluent.from_source([None], dims=["date"])
        result = faked_ensemble_transform(source, ens_num=1)
        assert isinstance(result, fluent.Action)

    def test_with_none_ens_num(self):
        from earthkit.workflows import fluent

        source = fluent.from_source([None], dims=["date"])
        result = faked_ensemble_transform(source, ens_num=None)
        assert isinstance(result, fluent.Action)


# --- _add_self_to_environment ---


class TestAddSelfToEnvironment:
    """Tests for _add_self_to_environment."""

    def test_dev_version_unchanged(self):
        """Development versions should not be added."""
        import earthkit.workflows.plugins.anemoi as pkg

        original = pkg.__version__
        pkg.__version__ = "0.3.1.dev0"
        try:
            env = ["anemoi-inference~=0.10"]
            result = _add_self_to_environment(env)
            assert result == ["anemoi-inference~=0.10"]
        finally:
            pkg.__version__ = original

    def test_empty_list_stays_empty(self):
        """Empty environment lists should remain empty."""
        result = _add_self_to_environment([])
        assert result == []

    def test_adds_self_to_list(self):
        """Should append self to a non-empty list."""
        import earthkit.workflows.plugins.anemoi as pkg

        original = pkg.__version__
        pkg.__version__ = "1.2.3"
        try:
            env = ["anemoi-inference~=0.10"]
            result = _add_self_to_environment(env)
            assert "earthkit-workflows-anemoi~=1.2.3" in result
        finally:
            pkg.__version__ = original

    def test_no_duplicate(self):
        """Should not add if already present."""
        env = ["earthkit-workflows-anemoi~=1.0.0"]
        result = _add_self_to_environment(env)
        assert sum(1 for e in result if e.startswith("earthkit-workflows-anemoi")) == 1

    def test_dict_environment(self):
        """Should handle dict environments."""
        import earthkit.workflows.plugins.anemoi as pkg

        original = pkg.__version__
        pkg.__version__ = "1.2.3"
        try:
            env = {"inference": ["anemoi-inference~=0.10"], "dataset": []}
            result = _add_self_to_environment(env)
            assert "earthkit-workflows-anemoi~=1.2.3" in result["inference"]
            assert result["dataset"] == []
        finally:
            pkg.__version__ = original


# --- crack_environment ---


class TestCrackEnvironment:
    """Tests for crack_environment."""

    def test_none_environment(self):
        result = crack_environment(None, ["inference", "dataset"])
        assert "inference" in result
        assert "dataset" in result

    def test_list_environment(self):
        env = ["anemoi-inference~=0.10"]
        result = crack_environment(env, ["inference", "dataset"])
        assert result["inference"] == result["dataset"]

    def test_dict_environment(self):
        env = {"inference": ["pkg1"], "dataset": ["pkg2"]}
        result = crack_environment(env, ["inference", "dataset"])
        assert "pkg1" in result["inference"]
        assert "pkg2" in result["dataset"]

    def test_dict_missing_key(self):
        env = {"inference": ["pkg1"]}
        result = crack_environment(env, ["inference", "dataset"])
        assert "pkg1" in result["inference"]
        assert isinstance(result["dataset"], list)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Invalid type"):
            crack_environment(42, ["inference"])


# --- expansion_qube_from_variables ---


class TestExpansionQubeFromVariables:
    """Tests for expansion_qube_from_variables."""

    @fake_checkpoints
    def test_basic(self):
        """Build a qube from variables extracted from a checkpoint."""
        from anemoi.inference.checkpoint import Checkpoint

        ckpt_path = Path(__file__).parent / "checkpoints" / "simple.yaml"
        ckpt = Checkpoint(ckpt_path)

        variables = ckpt._metadata.select_variables(include=["diagnostic", "prognostic"], has_mars_requests=False)
        variables_metadata = ckpt._metadata.typed_variables
        model_step = ckpt._metadata.timestep.seconds

        qube_dict = expansion_qube_from_variables({"data": variables}, variables_metadata, model_step, "1D")
        axes = qube_dict["data"].axes()
        assert "step" in axes
        assert "param" in axes

    @fake_checkpoints
    def test_matches_metadata_version(self):
        """expansion_qube_from_variables should produce same result as expansion_qube_from_metadata."""
        from anemoi.inference.checkpoint import Checkpoint

        ckpt_path = Path(__file__).parent / "checkpoints" / "simple.yaml"
        ckpt = Checkpoint(ckpt_path)

        variables = ckpt._metadata.select_variables(include=["diagnostic", "prognostic"], has_mars_requests=False)
        variables_metadata = ckpt._metadata.typed_variables
        model_step = ckpt._metadata.timestep.seconds

        qube_from_vars = expansion_qube_from_variables({"data": variables}, variables_metadata, model_step, "1D")
        qube_from_meta = expansion_qube_from_metadata(ckpt.multi_dataset_metadata, "1D")

        assert qube_from_vars["data"].axes() == qube_from_meta["data"].axes()


# --- expansion_qube_from_metadata with different lead times ---


class TestExpansionQubeLeadTimes:
    """Test expansion_qube_from_metadata with various lead time formats."""

    @fake_checkpoints
    def test_string_lead_time(self):
        from anemoi.inference.checkpoint import Checkpoint

        ckpt_path = Path(__file__).parent / "checkpoints" / "simple.yaml"
        qube_dict = expansion_qube_from_metadata(Checkpoint(ckpt_path).multi_dataset_metadata, "2D")
        qube = next(iter(qube_dict.values()))
        steps = qube.axes()["step"]
        assert max(steps) == 48

    @fake_checkpoints
    def test_full_atmo_checkpoint(self):
        """Test with full_atmo checkpoint that has pressure levels."""
        from anemoi.inference.checkpoint import Checkpoint

        ckpt_path = Path(__file__).parent / "checkpoints" / "full_atmo.yaml"
        qube_dict = expansion_qube_from_metadata(Checkpoint(ckpt_path).multi_dataset_metadata, "1D")
        qube = next(iter(qube_dict.values()))
        axes = qube.axes()
        assert "level" in axes
        assert "levtype" in axes
        assert "pl" in axes["levtype"]
