# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Per-member ENS initial conditions.

The real ``MarsInput.retrieve`` is used, only the final ``earthkit.data.from_source("mars", request)`` call is
intercepted, so the asserted requests are exactly what would be sent to MARS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import earthkit.data as ekd
import pytest
import yaml
from anemoi.inference.inputs.cutout import Cutout
from anemoi.inference.inputs.mars import MarsInput
from anemoi.inference.testing import fake_checkpoints

from earthkit.workflows.plugins.anemoi import inference
from earthkit.workflows.plugins.anemoi.inference import _get_initial_conditions

CHECKPOINT = Path(__file__).parent / "checkpoints" / "simple.yaml"

# Prognostics are 10u, 10v, 2t and msl; lsm is a constant forcing and z a dynamic forcing (see `_checkpoint`)
PROGNOSTIC, CONSTANT, DYNAMIC = "prognostic", "constant", "dynamic"
# Dates requested for the initial conditions at 2020-01-01 00:00 (multistep input of 2, 6 hourly)
DATES = [("2019-12-31", "1800"), ("2020-01-01", "0000")]


def _checkpoint(tmp_path: Path, mars: dict[str, Any]) -> str:
    """Write a copy of the simple fake checkpoint whose variables have `mars` in their checkpoint requests."""
    metadata = yaml.safe_load(CHECKPOINT.read_text())
    variables_metadata = metadata["dataset"]["variables_metadata"]
    for variable in variables_metadata.values():
        if "mars" in variable:
            variable["mars"].update(mars)
    # Not constant in time: z becomes a dynamic forcing
    variables_metadata["z"].pop("constant_in_time")
    (tmp_path / "ens.yaml").write_text(yaml.safe_dump(metadata))
    return str(tmp_path / "ens.ckpt")


def _category(request: dict) -> str:
    param = request["param"]
    if param == "lsm":
        return CONSTANT
    if param == "z":
        return DYNAMIC
    assert param == ["10u", "10v", "2t", "msl"], param
    return PROGNOSTIC


@fake_checkpoints
def _retrieve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_config: dict | str,
    number: int | None,
    checkpoint_mars: dict[str, Any],
) -> list[dict]:
    """Run `_get_initial_conditions` and return the final MARS requests, in the order they are issued."""
    config = {
        "checkpoint": _checkpoint(tmp_path, checkpoint_mars),
        "runner": "testing",
        "device": "cpu",
        "date": "2020-01-01",
        "input": input_config,
    }

    requests: list[dict] = []
    from_source = ekd.from_source

    def intercept(name, *args, **kwargs):
        if name != "mars":
            return from_source(name, *args, **kwargs)
        requests.append(args[0])
        return from_source("empty")

    def create_cutout_state(self, *, dates, **kwargs):
        for source in self.sources.values():
            source.create_input_state(dates=dates, **kwargs)
        return {"fields": {}}

    monkeypatch.setattr(ekd, "from_source", intercept)
    monkeypatch.setattr(MarsInput, "_create_input_state", lambda self, *args, **kwargs: {"fields": {}})
    monkeypatch.setattr(Cutout, "create_input_state", create_cutout_state)

    _get_initial_conditions(config, "2020-01-01T00:00", number)
    return requests


ANALYSIS = {"class": "od", "stream": "oper", "type": "an", "expver": "0001"}


@pytest.fixture
def retrieve(tmp_path, monkeypatch):
    def _(input_config, number, checkpoint_mars=ANALYSIS):
        return _retrieve(tmp_path, monkeypatch, input_config, number, checkpoint_mars)

    return _


def _by_category(requests: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {PROGNOSTIC: [], CONSTANT: [], DYNAMIC: []}
    for request in requests:
        result[_category(request)].append(request)
    return result


def _times(requests: list[dict]) -> list[tuple[str, str]]:
    times = []
    for request in requests:
        for time in [request["time"]] if isinstance(request["time"], str) else request["time"]:
            times.append((request["date"], time))
    return sorted(times)


def _assert_forcings_unchanged(requests: dict[str, list[dict]], **expected):
    """Constant and dynamic forcings keep the configured analysis request, whatever the member."""
    for category in (CONSTANT, DYNAMIC):
        assert requests[category], category
        for request in requests[category]:
            assert request["stream"] == "oper"
            assert request["type"] == "an"
            assert "number" not in request or request["number"] == expected.get("number"), request
            assert "step" not in request, request


MARS_INPUT = {"mars": {}}


class TestMemberNone:
    def test_requests_unchanged(self, retrieve):
        requests = _by_category(retrieve(MARS_INPUT, None))
        assert _times(requests[PROGNOSTIC]) == DATES
        for request in requests[PROGNOSTIC]:
            assert request["stream"] == "oper"
            assert request["type"] == "an"
            assert "number" not in request
            assert "step" not in request
        _assert_forcings_unchanged(requests)

    def test_configured_ens_request_kept(self, retrieve):
        config = {"mars": {"stream": "enfo", "type": "pf", "number": 5, "step": 0}}
        for request in _by_category(retrieve(config, None))[PROGNOSTIC]:
            assert (request["stream"], request["type"], request["number"], request["step"]) == ("enfo", "pf", 5, 0)


class TestControlMember:
    def test_prognostics_are_enfo_cf_step_0(self, retrieve):
        requests = _by_category(retrieve(MARS_INPUT, 0))
        assert _times(requests[PROGNOSTIC]) == DATES
        for request in requests[PROGNOSTIC]:
            assert request == {
                "class": "od",
                "stream": "enfo",
                "type": "cf",
                "expver": "0001",
                "levtype": "sfc",
                "param": ["10u", "10v", "2t", "msl"],
                "date": request["date"],
                "time": request["time"],
                "grid": "O96",
                "step": 0,
            }
        _assert_forcings_unchanged(requests)

    @pytest.mark.parametrize("where", ["checkpoint", "input config", "both"])
    def test_number_is_removed(self, retrieve, where):
        checkpoint_mars = {**ANALYSIS, "number": 7} if where != "input config" else ANALYSIS
        input_config = {"mars": {"number": 9} if where != "checkpoint" else {}}
        requests = _by_category(retrieve(input_config, 0, checkpoint_mars))
        assert requests[PROGNOSTIC]
        for request in requests[PROGNOSTIC]:
            assert "number" not in request
            assert (request["stream"], request["type"], request["step"]) == ("enfo", "cf", 0)


class TestPerturbedMember:
    def test_prognostics_are_enfo_pf_number_step_0(self, retrieve):
        requests = _by_category(retrieve(MARS_INPUT, 3))
        assert _times(requests[PROGNOSTIC]) == DATES
        for request in requests[PROGNOSTIC]:
            assert request == {
                "class": "od",
                "stream": "enfo",
                "type": "pf",
                "expver": "0001",
                "levtype": "sfc",
                "param": ["10u", "10v", "2t", "msl"],
                "date": request["date"],
                "time": request["time"],
                "grid": "O96",
                "step": 0,
                "number": 3,
            }
        _assert_forcings_unchanged(requests)

    def test_number_replaces_configured_and_checkpoint_numbers(self, retrieve):
        requests = _by_category(retrieve({"mars": {"number": 9}}, 3, {**ANALYSIS, "number": 7}))
        assert {r["number"] for r in requests[PROGNOSTIC]} == {3}
        # Forcings are not touched, including by a configured number
        assert {r["number"] for r in requests[CONSTANT] + requests[DYNAMIC]} == {9}


class TestOverrides:
    @pytest.mark.parametrize("number, type_", [(0, "cf"), (3, "pf")])
    def test_wins_over_configured_stream_and_type(self, retrieve, number, type_):
        config = {"mars": {"stream": "oper", "type": "fc", "class": "ai", "expver": "0002", "grid": [1, 1]}}
        requests = _by_category(retrieve(config, number, {**ANALYSIS, "stream": "scda", "type": "4v"}))
        for request in requests[PROGNOSTIC]:
            assert (request["stream"], request["type"], request["step"]) == ("enfo", type_, 0)
            # What is not part of the member selection is kept as configured
            assert request["class"] == "ai"
            assert request["expver"] == "0002"
            assert request["grid"] == [1, 1]
        for request in requests[CONSTANT] + requests[DYNAMIC]:
            assert (request["stream"], request["type"]) == ("oper", "fc")

    @pytest.mark.parametrize("number, type_", [(0, "cf"), (3, "pf")])
    def test_configured_step_does_not_shift_dates(self, retrieve, number, type_):
        config = {"mars": {"type": "fc", "step": 6}}
        requests = _by_category(retrieve(config, number))
        # Without a member, anemoi shifts the dates back by the configured step: the request is for the run
        # that is 6 hours older. With a member the step is 0 and the dates are the initial conditions' dates.
        assert _times(requests[PROGNOSTIC]) == DATES
        assert {r["step"] for r in requests[PROGNOSTIC]} == {0}
        assert {r["type"] for r in requests[PROGNOSTIC]} == {type_}

    def test_configured_step_is_kept_without_member(self, retrieve):
        requests = _by_category(retrieve({"mars": {"type": "fc", "step": 6}}, None))
        assert _times(requests[PROGNOSTIC]) == [("2019-12-31", "1200"), ("2019-12-31", "1800")]
        assert {r["step"] for r in requests[PROGNOSTIC]} == {6}

    @pytest.mark.parametrize("number, type_", [(0, "cf"), (3, "pf")])
    def test_applied_after_configured_patches(self, retrieve, number, type_):
        # Patches matching both the original and the ENS request: the member override must still win
        patches = [
            [{"param": ["10u", "10v", "2t", "msl"]}, {"stream": "oper", "type": "an", "number": 11, "step": 6}],
            [{"stream": "oper"}, {"type": "fc", "expver": "0003"}],
        ]
        config = {"mars": {"patches": patches}}
        requests = _by_category(retrieve(config, number))
        for request in requests[PROGNOSTIC]:
            assert (request["stream"], request["type"], request["step"]) == ("enfo", type_, 0)
            assert request.get("number") == (None if number == 0 else number)
            # A patch which only matches the original request is applied
            assert request["expver"] == "0003"

    def test_patches_still_apply_to_forcings(self, retrieve):
        patches = [[{"param": "lsm"}, {"expver": "0004"}]]
        requests = _by_category(retrieve({"mars": {"patches": patches}}, 3))
        assert {r["expver"] for r in requests[CONSTANT]} == {"0004"}
        assert {r["expver"] for r in requests[PROGNOSTIC]} == {"0001"}

    def test_no_leakage_between_calls(self, retrieve):
        first = _by_category(retrieve(MARS_INPUT, 3))
        second = _by_category(retrieve(MARS_INPUT, None))
        third = _by_category(retrieve(MARS_INPUT, 0))
        assert {(r["stream"], r["type"]) for r in first[PROGNOSTIC]} == {("enfo", "pf")}
        assert {(r["stream"], r["type"]) for r in second[PROGNOSTIC]} == {("oper", "an")}
        assert all("number" not in r and "step" not in r for r in second[PROGNOSTIC])
        assert {(r["stream"], r["type"]) for r in third[PROGNOSTIC]} == {("enfo", "cf")}

    def test_shared_config_is_not_mutated(self, tmp_path, monkeypatch):
        mars = {"step": 0, "type": "fc", "number": 9, "patches": [[{"stream": "oper"}, {"type": "fc"}]]}
        input_config = {"mars": mars}
        snapshot = yaml.safe_dump(input_config)
        _retrieve(tmp_path, monkeypatch, input_config, 3, ANALYSIS)
        assert yaml.safe_dump(input_config) == snapshot


class TestCutout:
    @staticmethod
    def _cutout(*sources: tuple[str, dict]) -> dict:
        return {"cutout": {"sources": [{name: {"mars": cfg, "mask": None}} for name, cfg in sources]}}

    @pytest.mark.parametrize("number, type_", [(0, "cf"), (3, "pf")])
    def test_every_source_gets_the_member(self, retrieve, number, type_):
        config = self._cutout(
            ("lam", {"expver": "0010", "stream": "oper", "type": "fc", "number": 9}),
            ("global", {"expver": "0020", "step": 6, "type": "fc"}),
            ("other", {"expver": "0030", "class": "ai"}),
        )
        requests = retrieve(config, number)
        prognostics = [r for r in requests if _category(r) == PROGNOSTIC]
        assert sorted({r["expver"] for r in prognostics}) == ["0010", "0020", "0030"]
        assert len(prognostics) == 3 * len(DATES)
        for request in prognostics:
            assert (request["stream"], request["type"], request["step"]) == ("enfo", type_, 0)
            assert request.get("number") == (None if number == 0 else number)
        assert {r["date"] + r["time"] for r in prognostics} == {d + t for d, t in DATES}

    def test_nested_cutout(self, retrieve):
        config = {
            "cutout": {
                "sources": [
                    {"lam": {"mars": {"expver": "0010"}, "mask": None}},
                    {
                        "inner": {
                            "cutout": {"sources": [{"deep": {"mars": {"expver": "0020", "number": 9}, "mask": None}}]},
                            "mask": None,
                        }
                    },
                ]
            }
        }
        requests = retrieve(config, 0)
        prognostics = [r for r in requests if _category(r) == PROGNOSTIC]
        assert sorted({r["expver"] for r in prognostics}) == ["0010", "0020"]
        for request in prognostics:
            assert (request["stream"], request["type"], request["step"]) == ("enfo", "cf", 0)
            assert "number" not in request

    def test_member_none_unchanged(self, retrieve):
        config = self._cutout(("lam", {"expver": "0010"}), ("global", {"expver": "0020"}))
        prognostics = [r for r in retrieve(config, None) if _category(r) == PROGNOSTIC]
        assert len(prognostics) == 2 * len(DATES)
        for request in prognostics:
            assert (request["stream"], request["type"]) == ("oper", "an")
            assert "number" not in request and "step" not in request

    def test_forcings_unchanged(self, retrieve):
        config = self._cutout(("lam", {"expver": "0010"}), ("global", {"expver": "0020"}))
        requests = _by_category(retrieve(config, 3))
        _assert_forcings_unchanged(requests)
        assert {r["expver"] for r in requests[CONSTANT]} == {"0010", "0020"}


class TestNonMarsInput:
    def test_warns_and_is_left_alone(self, retrieve, monkeypatch):
        log = MagicMock()
        monkeypatch.setattr(inference, "LOG", log)
        # Not a MarsInput: nothing to retrieve from MARS, the member cannot be applied
        assert retrieve("dummy", 3) == []
        assert log.warning.called
        assert "not a MarsInput" in log.warning.call_args.args[0]

    def test_no_warning_without_member(self, retrieve, monkeypatch):
        log = MagicMock()
        monkeypatch.setattr(inference, "LOG", log)
        retrieve("dummy", None)
        log.warning.assert_not_called()
