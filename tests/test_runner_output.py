# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests of the `output` option of the `CascadeRunner`, which always uses the cascade output."""

import logging
from pathlib import Path

import pytest
from anemoi.inference.testing import fake_checkpoints

from earthkit.workflows.plugins.anemoi.fluent import from_config
from earthkit.workflows.plugins.anemoi.runner import CascadeOutput, CascadeRunner

CKPT = (Path(__file__).parent / "checkpoints/simple.yaml").absolute()


def _assert_cascade_output(runner: CascadeRunner) -> None:
    assert runner.config.output == "cascade"
    assert all(isinstance(output, CascadeOutput) for output in runner.outputs.values())


@fake_checkpoints
def test_runner_from_dumped_run_configuration(mock_config, caplog):
    """A dumped `RunConfiguration` has `output`, which must not clash with the output of the runner."""
    dumped = mock_config.model_dump()
    assert "output" in dumped
    dumped["checkpoint"] = str(CKPT)

    with caplog.at_level(logging.WARNING):
        runner = CascadeRunner(**dumped)

    _assert_cascade_output(runner)
    assert "ignored" not in caplog.text


@fake_checkpoints
@pytest.mark.parametrize("output", ["cascade", {"cascade": {}}])
def test_runner_cascade_output_is_dropped_silently(output, caplog):
    with caplog.at_level(logging.WARNING):
        runner = CascadeRunner(str(CKPT), output=output, input="dummy", device="cpu")
    _assert_cascade_output(runner)
    assert "ignored" not in caplog.text


@fake_checkpoints
@pytest.mark.parametrize("output", ["grib", {"grib": {"path": "out.grib"}}])
def test_runner_warns_when_another_output_is_given(output, caplog):
    with caplog.at_level(logging.WARNING):
        runner = CascadeRunner(str(CKPT), output=output, input="dummy", device="cpu")
    assert "always uses the cascade output" in caplog.text
    _assert_cascade_output(runner)


@fake_checkpoints
def test_from_config_task_config_builds_a_runner(mock_config, monkeypatch):
    """The config `from_config` gives to the model task, which includes `output`, is accepted by `CascadeRunner`."""
    from earthkit.workflows.plugins.anemoi import fluent as fluent_module

    configs: list[dict] = []
    create_task_instance = fluent_module.fluent.create_task_instance

    def recording(*args, **kwargs):
        config = kwargs.get("static_input_kw", {}).get("config")
        if config is not None:
            configs.append(config)
        return create_task_instance(*args, **kwargs)

    monkeypatch.setattr(fluent_module.fluent, "create_task_instance", recording)

    from_config(mock_config, date="2020-01-01", lead_time="1D", checkpoint=str(CKPT), input="dummy")

    # The model task is the last created
    assert "output" in configs[-1]
    _assert_cascade_output(CascadeRunner(**configs[-1]))
