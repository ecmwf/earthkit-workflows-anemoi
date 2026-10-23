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
from earthkit.workflows.metadata import Requirements

from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME, STAGES
from earthkit.workflows.plugins.anemoi.utils import (
    _add_self_to_environment,
    crack_environment,
    crack_requirements,
    expansion_qube_from_metadata,
    expansion_qube_from_variables,
    expose_ensemble_dimension,
    faked_ensemble_transform,
    parse_ensemble_members,
    warn_environment_deprecated,
)

# --- parse_ensemble_members ---


class TestParseEnsembleMembers:
    """Tests for parse_ensemble_members."""

    def test_none_returns_list_of_none(self):
        assert parse_ensemble_members(None) == [None]

    def test_single_int_is_member_id(self):
        result = parse_ensemble_members(3)
        assert result == [3]

    def test_single_int_one(self):
        result = parse_ensemble_members(1)
        assert result == [1]

    def test_zero_is_control_member(self):
        assert parse_ensemble_members(0) == [0]

    def test_sequence_passed_through(self):
        result = parse_ensemble_members([5, 10, 15])
        assert result == [5, 10, 15]

    def test_range_passed_through(self):
        result = parse_ensemble_members(range(1, 4))
        assert result == [1, 2, 3]

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must not be negative"):
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

    def test_does_not_mutate_input_dict(self):
        """The function should return a new dict and leave the input unchanged."""
        state = {"data": {"date": "2020-01-01"}}
        result = expose_ensemble_dimension(state, 1)
        assert result is not state
        assert result["data"] is not state["data"]
        assert state == {"data": {"date": "2020-01-01"}}

    def test_non_dict_raises(self):
        with pytest.raises(AssertionError, match="dictionary"):
            expose_ensemble_dimension("not a dict", 1)

    def test_none_placeholder_passed_through(self):
        """`None` initial conditions are a graph placeholder, there is no state to label."""
        assert expose_ensemble_dimension(None, 3) is None
        assert expose_ensemble_dimension(None, None) is None

    def test_uses_constant_not_hardcoded(self):
        """Ensure the function uses the ENSEMBLE_DIMENSION_NAME constant."""
        state = {"data": {}}
        result = expose_ensemble_dimension(state, 42)
        assert ENSEMBLE_DIMENSION_NAME in result["data"]
        # If the constant were still 'ensemble_member', this would trivially pass.
        # The important test is that only ENSEMBLE_DIMENSION_NAME is set.
        assert len([k for k in result["data"] if k in ("ensemble_member", "number")]) == 1


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

    def test_list_environment_not_aliased(self, monkeypatch):
        """Each key gets its own list and the caller's list is untouched."""
        import earthkit.workflows.plugins.anemoi as pkg

        monkeypatch.setattr(pkg, "__version__", "1.2.3")
        env = ["pkg1"]
        result = crack_environment(env, ["inference", "dataset"])
        assert result["inference"] is not result["dataset"]
        assert result["inference"] is not env
        assert env == ["pkg1"]
        assert set(result["inference"]) == {"pkg1", "earthkit-workflows-anemoi~=1.2.3"}
        result["inference"].append("other")
        assert "other" not in result["dataset"]

    def test_dict_environment_not_mutated(self, monkeypatch):
        import earthkit.workflows.plugins.anemoi as pkg

        monkeypatch.setattr(pkg, "__version__", "1.2.3")
        env = {"inference": ["pkg1"], "dataset": ["pkg2"]}
        result = crack_environment(env, ["inference", "dataset"])
        assert env == {"inference": ["pkg1"], "dataset": ["pkg2"]}
        assert result["inference"] is not env["inference"]


class TestAddSelfNoMutation:
    """_add_self_to_environment must not modify or return its input."""

    def test_list_not_mutated(self, monkeypatch):
        import earthkit.workflows.plugins.anemoi as pkg

        monkeypatch.setattr(pkg, "__version__", "1.2.3")
        env = ["pkg1"]
        result = _add_self_to_environment(env)
        assert env == ["pkg1"]
        assert result == ["pkg1", "earthkit-workflows-anemoi~=1.2.3"]

    def test_existing_self_returns_copy(self):
        env = ["earthkit-workflows-anemoi==1.0.0"]
        result = _add_self_to_environment(env)
        assert result == env
        assert result is not env

    def test_dict_not_mutated(self, monkeypatch):
        import earthkit.workflows.plugins.anemoi as pkg

        monkeypatch.setattr(pkg, "__version__", "1.2.3")
        env = {"inference": ["pkg1"], "dataset": []}
        result = _add_self_to_environment(env)
        assert env == {"inference": ["pkg1"], "dataset": []}
        assert result["inference"] == ["pkg1", "earthkit-workflows-anemoi~=1.2.3"]
        assert result["dataset"] == []

    def test_dev_version_returns_copy(self, monkeypatch):
        import earthkit.workflows.plugins.anemoi as pkg

        monkeypatch.setattr(pkg, "__version__", "0.3.1.dev0")
        env = ["pkg1"]
        result = _add_self_to_environment(env)
        assert result == env
        assert result is not env


# --- warn_environment_deprecated ---


class TestWarnEnvironmentDeprecated:
    """Tests for warn_environment_deprecated."""

    def test_warns_with_function_name(self):
        with pytest.warns(DeprecationWarning, match=r"`my_func`.*requirements=") as record:
            warn_environment_deprecated("my_func")
        assert len(record) == 1

    def test_default_stacklevel_points_at_caller_of_caller(self):
        def public_function():
            warn_environment_deprecated("public_function")

        with pytest.warns(DeprecationWarning) as record:
            public_function()
        assert Path(record[0].filename) == Path(__file__)


# --- crack_requirements ---


@pytest.fixture
def pinned_version(monkeypatch):
    """Pin the package version so the self-pin does not depend on a dev version."""
    import earthkit.workflows.plugins.anemoi as pkg

    monkeypatch.setattr(pkg, "__version__", "1.2.3")
    return "earthkit-workflows-anemoi~=1.2.3"


KEYS = ["inference", "initial_conditions"]


class TestCrackRequirements:
    """Tests for crack_requirements."""

    def test_stages(self):
        assert STAGES == ("inference", "initial_conditions")

    def test_none(self):
        result = crack_requirements(None, KEYS)
        assert set(result) == set(KEYS)
        for req in result.values():
            assert req == Requirements()

    def test_empty_dict(self):
        result = crack_requirements({}, KEYS)
        assert all(req == Requirements() for req in result.values())

    def test_single_requirements_broadcast(self, pinned_version):
        result = crack_requirements(Requirements(environment=["pkg1"], needs_gpu=True), KEYS)
        for req in result.values():
            assert set(req.environment) == {"pkg1", pinned_version}
            assert req.needs_gpu is True
        assert result["inference"] is not result["initial_conditions"]

    def test_plain_dict_of_fields(self, pinned_version):
        result = crack_requirements({"environment": ["pkg1"], "needs_gpu": True}, KEYS)
        for req in result.values():
            assert isinstance(req, Requirements)
            assert set(req.environment) == {"pkg1", pinned_version}
            assert req.needs_gpu is True

    def test_plain_dict_single_field(self):
        result = crack_requirements({"needs_gpu": True}, KEYS)
        for req in result.values():
            assert req.needs_gpu is True
            assert req.environment == []

    def test_per_stage_with_requirements_and_dicts(self, pinned_version):
        result = crack_requirements(
            {
                "inference": Requirements(needs_gpu=True, environment=["pkg1"]),
                "initial_conditions": {"environment": ["pkg2"]},
            },
            KEYS,
        )
        assert set(result["inference"].environment) == {"pkg1", pinned_version}
        assert result["inference"].needs_gpu is True
        assert set(result["initial_conditions"].environment) == {"pkg2", pinned_version}
        assert result["initial_conditions"].needs_gpu is None

    def test_missing_stage_key_is_empty(self):
        result = crack_requirements({"inference": {"needs_gpu": True}}, KEYS)
        assert result["inference"].needs_gpu is True
        assert result["initial_conditions"] == Requirements()

    def test_extra_valid_stage_ignored(self):
        result = crack_requirements(
            {"inference": {"needs_gpu": True}, "initial_conditions": {"environment": ["x"]}}, ["inference"]
        )
        assert set(result) == {"inference"}
        assert result["inference"].needs_gpu is True
        assert result["inference"].environment == []

    def test_dataset_stage_is_invalid(self):
        """Dataset creation was removed, so `dataset` is not a stage any more."""
        with pytest.raises(ValueError, match="Invalid keys"):
            crack_requirements({"inference": {"needs_gpu": True}, "dataset": {"environment": ["x"]}}, KEYS)
        with pytest.raises(ValueError, match="Invalid keys"):
            crack_requirements({"dataset": {"environment": ["x"]}}, KEYS)

    def test_deprecated_environment_dict_ignores_other_keys(self, pinned_version):
        result = crack_requirements(None, KEYS, environment={"inference": ["pkg1"], "dataset": ["pkg2"]})
        assert set(result["inference"].environment) == {"pkg1", pinned_version}
        assert result["initial_conditions"].environment == []

    @pytest.mark.parametrize(
        "bad",
        [
            {"unknown": ["pkg"]},
            {"inference": {}, "unknown": {}},
            {"environment": ["pkg"], "inference": {}},
            {"needs_gpu": True, "initial_conditions": {}},
            {"environment": ["pkg"], "unknown": 1},
        ],
    )
    def test_unknown_or_mixed_keys_raise(self, bad):
        with pytest.raises(ValueError, match="Invalid keys"):
            crack_requirements(bad, KEYS)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Invalid type for requirements"):
            crack_requirements(["pkg"], KEYS)

    def test_invalid_stage_value_type_raises(self):
        with pytest.raises(TypeError, match="requirements\\['inference'\\]"):
            crack_requirements({"inference": ["pkg"]}, KEYS)

    def test_invalid_field_in_stage_dict_raises(self):
        with pytest.raises(ValueError):
            crack_requirements({"inference": {"bogus": 1}}, KEYS)

    def test_merge_deprecated_list_environment(self, pinned_version):
        result = crack_requirements(Requirements(environment=["pkg1"]), KEYS, environment=["pkg2", "pkg1"])
        for req in result.values():
            assert set(req.environment) == {"pkg1", "pkg2", pinned_version}
            assert len(req.environment) == 3

    def test_merge_deprecated_dict_environment(self, pinned_version):
        result = crack_requirements(
            {"inference": {"environment": ["pkg1"]}},
            KEYS,
            environment={"inference": ["pkg2"], "initial_conditions": ["pkg3"]},
        )
        assert set(result["inference"].environment) == {"pkg1", "pkg2", pinned_version}
        assert set(result["initial_conditions"].environment) == {"pkg3", pinned_version}

    def test_deprecated_environment_only(self, pinned_version):
        result = crack_requirements(None, KEYS, environment=["pkg1"])
        for req in result.values():
            assert set(req.environment) == {"pkg1", pinned_version}

    def test_deprecated_environment_invalid_type(self):
        with pytest.raises(TypeError, match="Invalid type for environment"):
            crack_requirements(None, KEYS, environment=42)

    @pytest.mark.parametrize("needs_gpu", [True, False, None])
    def test_needs_gpu_kept_and_merged_with_environment(self, needs_gpu):
        """The deprecated environment adds packages only, so it never changes needs_gpu.

        The merge is the same as `update_requirements` (`other or into`), with `other` carrying
        no `needs_gpu`.
        """
        from earthkit.workflows.metadata import update_requirements

        result = crack_requirements(Requirements(needs_gpu=needs_gpu), ["inference"], environment=["pkg1"])
        expected = Requirements(needs_gpu=needs_gpu)
        update_requirements(expected, Requirements(environment=["pkg1"]))
        assert result["inference"].needs_gpu is expected.needs_gpu is needs_gpu

    def test_needs_gpu_per_stage(self):
        result = crack_requirements(
            {"inference": {"needs_gpu": True}, "initial_conditions": {"needs_gpu": False}}, KEYS
        )
        assert result["inference"].needs_gpu is True
        assert result["initial_conditions"].needs_gpu is False

    def test_environment_merge_matches_upstream(self, pinned_version):
        from earthkit.workflows.metadata import update_requirements

        into = Requirements(environment=["a", "b"])
        update_requirements(into, Requirements(environment=["b", "c"]))
        result = crack_requirements(Requirements(environment=["a", "b"]), ["inference"], environment=["b", "c"])
        assert set(result["inference"].environment) - {pinned_version} == set(into.environment)

    def test_self_pin_added_once(self, pinned_version):
        result = crack_requirements(Requirements(environment=["pkg1"]), KEYS, environment=["pkg2"])
        for req in result.values():
            assert req.environment.count(pinned_version) == 1

    def test_self_pin_not_added_to_empty(self, pinned_version):
        result = crack_requirements({"inference": {"environment": ["pkg1"]}}, KEYS)
        assert pinned_version in result["inference"].environment
        assert result["initial_conditions"].environment == []

    def test_existing_self_entry_suppresses_pin(self, pinned_version):
        result = crack_requirements(
            Requirements(environment=["pkg1"]), KEYS, environment=["earthkit-workflows-anemoi==0.9.0"]
        )
        for req in result.values():
            assert set(req.environment) == {"pkg1", "earthkit-workflows-anemoi==0.9.0"}

    def test_dev_version_no_pin(self, monkeypatch):
        import earthkit.workflows.plugins.anemoi as pkg

        monkeypatch.setattr(pkg, "__version__", "0.3.1.dev0")
        result = crack_requirements({"environment": ["pkg1"]}, KEYS)
        for req in result.values():
            assert req.environment == ["pkg1"]

    def test_inputs_not_mutated(self, pinned_version):
        single = Requirements(environment=["pkg1"], needs_gpu=True)
        per_stage = {
            "inference": Requirements(environment=["pkg1"]),
            "initial_conditions": {"environment": ["pkg2"]},
        }
        fields = {"environment": ["pkg3"], "needs_gpu": False}
        env_list = ["pkg4"]
        env_dict = {"inference": ["pkg5"]}

        for req in (single, per_stage, fields):
            crack_requirements(req, KEYS, environment=env_list)
        crack_requirements(per_stage, KEYS, environment=env_dict)

        assert single == Requirements(environment=["pkg1"], needs_gpu=True)
        assert per_stage["inference"] == Requirements(environment=["pkg1"])
        assert per_stage["initial_conditions"] == {"environment": ["pkg2"]}
        assert fields == {"environment": ["pkg3"], "needs_gpu": False}
        assert env_list == ["pkg4"]
        assert env_dict == {"inference": ["pkg5"]}

    def test_reused_requirements_not_shared_or_mutated(self, pinned_version):
        shared = Requirements(environment=["pkg1"])
        first = crack_requirements(shared, KEYS, environment=["pkg2"])
        second = crack_requirements(shared, KEYS)
        assert shared == Requirements(environment=["pkg1"])
        assert set(second["inference"].environment) == {"pkg1", pinned_version}
        objects = [*first.values(), *second.values()]
        assert len({id(o) for o in objects}) == 4
        assert all(o is not shared for o in objects)
        first["inference"].environment.append("changed")
        assert "changed" not in first["initial_conditions"].environment
        assert "changed" not in second["inference"].environment
        assert shared.environment == ["pkg1"]


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

        qube_dict = expansion_qube_from_variables({"data": variables}, {"data": variables_metadata}, model_step, "1D")
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

        qube_from_vars = expansion_qube_from_variables(
            {"data": variables}, {"data": variables_metadata}, model_step, "1D"
        )
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
