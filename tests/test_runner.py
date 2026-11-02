# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os

from anemoi.inference.runner import Runner

from earthkit.workflows.plugins.anemoi.runner import CascadeRunner, _grib_metadata


def test_cascade_runner_applies_env_before_runner_setup(monkeypatch, tmp_path):
    # Register the variable with monkeypatch so it is restored on teardown
    monkeypatch.setenv("ANEMOI_INFERENCE_NUM_CHUNKS", "unset")
    monkeypatch.delenv("ANEMOI_INFERENCE_NUM_CHUNKS")

    seen = {}

    def fake_init(self, config):
        # The environment must already be set when the base runner is set up
        seen["value"] = os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS")

    monkeypatch.setattr(Runner, "__init__", fake_init)

    CascadeRunner(checkpoint=str(tmp_path / "dummy.ckpt"), env={"ANEMOI_INFERENCE_NUM_CHUNKS": 16})

    assert seen["value"] == "16"
    assert os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] == "16"


def test_grib_metadata_no_member():
    assert _grib_metadata(None) == {"edition": 2, "type": "fc", "stream": "oper", "class": "ai"}


def test_grib_metadata_control_member():
    assert _grib_metadata(0) == {
        "edition": 2,
        "type": "cf",
        "class": "ai",
        "productDefinitionTemplateNumber": 1,
        "stream": "enfo",
        "number": 0,
    }


def test_grib_metadata_perturbed_member():
    assert _grib_metadata(3) == {
        "edition": 2,
        "type": "pf",
        "class": "ai",
        "productDefinitionTemplateNumber": 1,
        "stream": "enfo",
        "number": 3,
    }
