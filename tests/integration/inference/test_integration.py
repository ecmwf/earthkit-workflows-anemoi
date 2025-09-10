# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.testing import save_fake_checkpoint
from anemoi.transform.variables.variables import VariableFromMarsVocabulary
from anemoi.utils.testing import GetTestData
from omegaconf import OmegaConf

import earthkit.workflows.plugins.anemoi.fluent as wa_fluent
from earthkit.workflows.plugins.anemoi.testing import testing_registry
from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

from ...executor import run_job

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class Setup(NamedTuple):
    """Integration test setup parameters."""

    checkpoint: Path
    inference_config: dict
    workflows_config: dict
    checks: list[dict]


INTEGRATION_ROOT = Path(__file__).resolve().parent

# each model has its own folder in the integration tests directory
# and contains at least a metadata.json and config.yaml file
MODELS = [
    path.name
    for path in INTEGRATION_ROOT.iterdir()
    if path.is_dir() and (path / "metadata.json").exists() and (path / "config.yaml").exists()
]

# each model can have more than one test configuration, defined as a listconfig in config.yaml
# the integration test is parameterised over the models and their test configurations
MODEL_CONFIGS = [
    (model, config) for model in MODELS for config in OmegaConf.load(INTEGRATION_ROOT / model / "config.yaml")
]


def _inject_provenance(metadata: dict) -> dict:
    """Inject provenance information into metadata dictionary."""
    metadata = metadata.copy()
    from anemoi.utils.provenance import gather_provenance_info

    metadata["provenance_training"] = gather_provenance_info(full=False)
    return metadata


@pytest.fixture(params=MODEL_CONFIGS, ids=[f"{model}/{config.name}" for model, config in MODEL_CONFIGS])
def test_setup(request, get_test_data: GetTestData, tmp_path: Path) -> Setup:
    model, config = request.param
    workflows_config = config.workflows_config
    inference_config = config.inference_config
    s3_path = f"anemoi-integration-tests/inference/{model}"

    # prepare checkpoint
    checkpoint_path = tmp_path / Path("checkpoint.ckpt")
    with open(INTEGRATION_ROOT / model / "metadata.json") as f:
        metadata = json.load(f)

    metadata = _inject_provenance(metadata)

    supporting_arrays = {}
    if array_names := list(metadata.get("supporting_arrays_paths", {})):

        def load_array(name):
            return name, np.load(get_test_data(f"{s3_path}/supporting-arrays/{name}.npy"))

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load_array, array_names))

        supporting_arrays = dict(results)
        LOG.info(f"Supporting arrays: {supporting_arrays.keys()}")

    save_fake_checkpoint(metadata, checkpoint_path, supporting_arrays=supporting_arrays)

    # substitute inference config with real paths
    # OmegaConf.register_new_resolver("input", lambda i=0: str(input_data[i]), replace=True)
    # OmegaConf.register_new_resolver("output", lambda: str(output), replace=True)
    # OmegaConf.register_new_resolver("checkpoint", lambda: str(checkpoint_path), replace=True)
    # OmegaConf.register_new_resolver("s3", lambda: str(f"{TEST_DATA_URL}{s3_path}"), replace=True)

    # save the inference config to disk
    inference_config = OmegaConf.to_object(inference_config)
    workflows_config = OmegaConf.to_object(workflows_config)

    LOG.info(f"Resolved inference config:\n{inference_config}")
    LOG.info(f"Resolved workflows config:\n{workflows_config}")

    assert isinstance(inference_config, dict), type(inference_config)
    assert isinstance(workflows_config, dict), type(workflows_config)

    # with open(tmp_path / "integration_test.yaml", "w") as f:
    #     f.write(inference_config)

    return Setup(
        checkpoint=checkpoint_path,
        inference_config=inference_config,
        workflows_config=workflows_config,
        checks=config.checks,
    )


def _typed_variables_output(checkpoint: Checkpoint) -> list[VariableFromMarsVocabulary]:
    output_variables = checkpoint.output_tensor_index_to_variable.values()
    return [checkpoint.typed_variables[name] for name in output_variables]  # type: ignore


def test_integration(test_setup: Setup, shared_gateway: str) -> None:
    ckpt, inf_conf, work_conf, checks = test_setup
    LOG.info(
        f"Running integration test with checkpoint: {ckpt}, inference_config: {inf_conf}, workflows_config: {work_conf}"
    )

    work_conf.setdefault("input", "dummy")
    work_conf.setdefault("date", "20250101T00")
    work_conf.setdefault("lead_time", 72)

    action = wa_fluent.from_input(
        ckpt=ckpt,
        **work_conf,
        **inf_conf,
    )
    action = action.concatenate("param").concatenate("step")

    if ENSEMBLE_DIMENSION_NAME in action.nodes.dims:
        action = action.concatenate(ENSEMBLE_DIMENSION_NAME)

    result = run_job(action, url=shared_gateway, tries=60)
    assert result, "The job didn't return anything"

    checkpoint = Checkpoint(str(ckpt))
    checkpoint_output_variables = _typed_variables_output(checkpoint)
    LOG.info(f"Checkpoint output variables: {checkpoint_output_variables}")

    # run the checks defined in the test configuration
    # checks is a list of dicts, each with a single key-value pair
    for checks in checks:
        check, kwargs = next(iter(checks.items()))
        # config can optionally pass expected output variables, by default it uses the checkpoint variables
        expected_variables_config = kwargs.pop("expected_variables", [])
        expected_variables = [
            VariableFromMarsVocabulary(var, {"param": var}) for var in expected_variables_config
        ] or checkpoint_output_variables

        testing_registry.create(
            check,
            fieldlist=result,
            expected_variables=expected_variables,
            checkpoint=checkpoint,
            **kwargs,
        )
