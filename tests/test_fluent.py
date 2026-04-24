# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from anemoi.inference.testing import fake_checkpoints
from earthkit.workflows.fluent import nodetree_arrays
from xarray import DataArray, DataTree

from earthkit.workflows.plugins.anemoi.fluent import Action, Inference, from_config, from_initial_conditions, from_input
from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

STANDARD_INFERENCE_TESTS = [
    # Test inputs of ensembles
    [
        "simple",
        2,
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 2, "param": 6, "date": 1},
    ],
    [
        "simple",
        8,
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 8, "param": 6, "date": 1},
    ],
    [
        "simple",
        8,
        {"date": "2020-01-01", "lead_time": "4D"},
        {"step": 16, ENSEMBLE_DIMENSION_NAME: 8, "param": 6, "date": 1},
    ],
    # Test different model configs
    [
        "simple",
        2,
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 2, "param": 6, "date": 1},
    ],
    [
        "surface",
        2,
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 2, "param": 6, "date": 1},
    ],
    [
        "full_atmo",
        2,
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 2, "param": 6, "date": 1, "level": 1},
    ],
]


def assert_shape(action: Action, shape: dict[str, int]):
    """Assert action nodes are of the correct shape"""
    shapes = defaultdict(set)

    def count(node: DataTree | DataArray):
        for dim, size in node.coords.items():
            shapes[dim].update(np.atleast_1d(size.values))

        if isinstance(node, DataTree):
            for child in node.children:
                count(node[child])

    count(action.nodes)

    for dim in shape:
        assert dim in shapes
        assert len(shapes[dim]) == shape[dim]


@pytest.mark.parametrize("ckpt, ensemble_members, kwargs, shape", STANDARD_INFERENCE_TESTS)
@fake_checkpoints
def test_from_input(ckpt, ensemble_members, kwargs, shape):
    """Test running from initial conditions"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()

    action = from_input(ckpt_full_path, "dummy", ensemble_members=ensemble_members, **kwargs)
    assert_shape(action, shape)


@pytest.mark.parametrize("ckpt, ensemble_members, kwargs, shape", STANDARD_INFERENCE_TESTS)
@fake_checkpoints
def test_inference_from_input(ckpt, ensemble_members, kwargs, shape):
    """Test running from input using the class API"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()

    inference = Inference(ckpt_full_path, lead_time=kwargs["lead_time"])
    action = inference.from_input("dummy", kwargs["date"], ensemble_members=ensemble_members)
    assert_shape(action, shape)


@pytest.mark.parametrize(
    "ckpt, ensemble_members, kwargs, shape",
    STANDARD_INFERENCE_TESTS,
)
@fake_checkpoints
def test_from_config(mock_config, ckpt, ensemble_members, kwargs, shape):
    """Test running from initial conditions"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()

    action = from_config(
        mock_config, ensemble_members=ensemble_members, **kwargs, checkpoint=str(ckpt_full_path), input="dummy"
    )
    assert_shape(action, shape)


@pytest.mark.parametrize(
    "ckpt, ensemble_members, kwargs, shape",
    STANDARD_INFERENCE_TESTS,
)
@fake_checkpoints
def test_from_initial_conditions_from_none(ckpt, ensemble_members, kwargs, shape):
    """Test running from initial conditions"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()
    kwargs = kwargs.copy()
    shape = shape.copy()
    kwargs.pop("date", None)

    action = from_initial_conditions(ckpt_full_path, None, ensemble_members=ensemble_members, **kwargs)
    shape.pop("date", None)
    assert_shape(action, shape)


@pytest.mark.parametrize(
    "ckpt, ensemble_members, kwargs, shape",
    STANDARD_INFERENCE_TESTS,
)
@fake_checkpoints
def test_inference_from_initial_conditions_from_none(ckpt, ensemble_members, kwargs, shape):
    """Test running from initial conditions using the class API"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()
    kwargs = kwargs.copy()
    shape = shape.copy()
    kwargs.pop("date", None)

    inference = Inference(ckpt_full_path, lead_time=kwargs.pop("lead_time"))
    action = inference.from_initial_conditions(None, ensemble_members=ensemble_members, **kwargs)
    shape.pop("date", None)
    assert_shape(action, shape)


@pytest.mark.parametrize(
    "ckpt, ensemble_members, kwargs, shape",
    STANDARD_INFERENCE_TESTS,
)
@fake_checkpoints
def test_from_initial_conditions_with_no_checkpoint_file(ckpt, ensemble_members, kwargs, shape):
    """Test running with no checkpoint file"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()
    kwargs = kwargs.copy()
    shape = shape.copy()

    from anemoi.inference.checkpoint import Checkpoint

    metadata = Checkpoint(ckpt_full_path)._metadata  # type: ignore

    kwargs.pop("date", None)

    action = from_initial_conditions(
        "non_existent_checkpoint.ckpt", None, ensemble_members=ensemble_members, metadata=metadata, **kwargs
    )
    shape.pop("date", None)
    assert_shape(action, shape)


@pytest.mark.parametrize(
    "ckpt, ensemble_members, kwargs, shape",
    STANDARD_INFERENCE_TESTS,
)
@fake_checkpoints
def test_from_initial_conditions_from_action(ckpt, ensemble_members, kwargs, shape):
    """Test running from initial conditions"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()
    kwargs = kwargs.copy()
    shape = shape.copy()
    kwargs.pop("date", None)

    from earthkit.workflows import fluent

    init_conditions = fluent.from_source(
        [None for _ in range(ensemble_members)],
        dims=[ENSEMBLE_DIMENSION_NAME],
        coords={ENSEMBLE_DIMENSION_NAME: range(ensemble_members)},
    )
    shape.pop("date", None)

    action = from_initial_conditions(ckpt_full_path, init_conditions, **kwargs)
    assert_shape(action, shape)


@pytest.mark.parametrize(
    "ckpt, ensemble_members, kwargs, shape",
    STANDARD_INFERENCE_TESTS,
)
@fake_checkpoints
def test_from_initial_conditions_from_infer(ckpt, ensemble_members, kwargs, shape):
    """Test running from initial conditions"""
    ckpt_full_path = (Path(__file__).parent / f"checkpoints/{ckpt}.yaml").absolute()
    kwargs = kwargs.copy()
    shape = shape.copy()
    kwargs.pop("date", None)

    from earthkit.workflows import fluent

    init_conditions = Action(
        fluent.from_source(
            [None for _ in range(ensemble_members)],
            dims=[ENSEMBLE_DIMENSION_NAME],
            coords={ENSEMBLE_DIMENSION_NAME: range(ensemble_members)},
        ).nodes
    )
    shape.pop("date", None)

    action = init_conditions.infer(ckpt_full_path, **kwargs)
    assert_shape(action, shape)


# --- payload_metadata propagation ---

PAYLOAD_METADATA = {"source": "test", "run_id": "abc123"}
SIMPLE_CKPT = "simple"
SIMPLE_KWARGS = {"date": "2020-01-01", "lead_time": "1D"}


def assert_payload_metadata(action: Action, expected: dict) -> None:
    """Assert every node in action carries the expected metadata entries."""
    for _, narray in nodetree_arrays(action.nodes):
        for node in np.atleast_1d(narray.values).flatten():
            for key, value in expected.items():
                assert node.payload.metadata.get(key) == value, (
                    f"Node {node.name!r} missing metadata {key!r}={value!r}, " f"got {node.payload.metadata}"
                )


@fake_checkpoints
def test_from_input_propagates_payload_metadata():
    """payload_metadata passed to from_input is stored on every result node."""
    ckpt = (Path(__file__).parent / f"checkpoints/{SIMPLE_CKPT}.yaml").absolute()
    action = from_input(ckpt, "dummy", payload_metadata=PAYLOAD_METADATA, **SIMPLE_KWARGS)
    assert_payload_metadata(action, PAYLOAD_METADATA)


@fake_checkpoints
def test_from_initial_conditions_propagates_payload_metadata():
    """payload_metadata passed to from_initial_conditions is stored on every result node."""
    ckpt = (Path(__file__).parent / f"checkpoints/{SIMPLE_CKPT}.yaml").absolute()
    kwargs = SIMPLE_KWARGS.copy()
    kwargs.pop("date")
    action = from_initial_conditions(ckpt, None, payload_metadata=PAYLOAD_METADATA, **kwargs)
    assert_payload_metadata(action, PAYLOAD_METADATA)


@fake_checkpoints
def test_from_config_propagates_payload_metadata(mock_config):
    """payload_metadata passed to from_config is stored on every result node."""
    ckpt = (Path(__file__).parent / f"checkpoints/{SIMPLE_CKPT}.yaml").absolute()
    action = from_config(
        mock_config,
        payload_metadata=PAYLOAD_METADATA,
        checkpoint=str(ckpt),
        input="dummy",
        **SIMPLE_KWARGS,
    )
    assert_payload_metadata(action, PAYLOAD_METADATA)


@fake_checkpoints
def test_inference_from_input_propagates_payload_metadata():
    """payload_metadata passed to Inference.from_input is stored on every result node."""
    ckpt = (Path(__file__).parent / f"checkpoints/{SIMPLE_CKPT}.yaml").absolute()
    inference = Inference(ckpt, lead_time=SIMPLE_KWARGS["lead_time"])
    action = inference.from_input("dummy", SIMPLE_KWARGS["date"], payload_metadata=PAYLOAD_METADATA)
    assert_payload_metadata(action, PAYLOAD_METADATA)


@fake_checkpoints
def test_inference_from_initial_conditions_propagates_payload_metadata():
    """payload_metadata passed to Inference.from_initial_conditions is stored on every result node."""
    ckpt = (Path(__file__).parent / f"checkpoints/{SIMPLE_CKPT}.yaml").absolute()
    inference = Inference(ckpt, lead_time=SIMPLE_KWARGS["lead_time"])
    action = inference.from_initial_conditions(None, payload_metadata=PAYLOAD_METADATA)
    assert_payload_metadata(action, PAYLOAD_METADATA)


# --- dict metadata support ---


@pytest.fixture
def dict_metadata() -> dict:
    """Build a minimal metadata dict from the simple checkpoint YAML."""
    import yaml

    parent_dir = Path(__file__).parent
    with open(parent_dir / "checkpoints" / "simple.yaml") as f:
        raw = yaml.safe_load(f)
    return {
        "config": {
            "data": raw["config"]["data"],
        },
        "data_indices": raw["data_indices"],
        "dataset": {
            "variables": raw["dataset"]["variables"],
            "variables_metadata": raw["dataset"]["variables_metadata"],
        },
    }


def test_from_initial_conditions_with_dict_metadata(dict_metadata: dict) -> None:
    """Test running from initial conditions with dict metadata (no checkpoint file needed)."""
    action = from_initial_conditions(
        "non_existent.ckpt", None, metadata=dict_metadata, date="2020-01-01", lead_time="1D"
    )
    assert_shape(action, {"step": 4, ENSEMBLE_DIMENSION_NAME: 1, "param": 6})


def test_from_input_with_dict_metadata(dict_metadata: dict) -> None:
    """Test running from input with dict metadata (no checkpoint file needed)."""
    action = from_input("non_existent.ckpt", "dummy", metadata=dict_metadata, date="2020-01-01", lead_time="1D")
    assert_shape(action, {"step": 4, ENSEMBLE_DIMENSION_NAME: 1, "param": 6, "date": 1})


@fake_checkpoints
def test_inference_class_with_dict_metadata(dict_metadata: dict) -> None:
    """Test Inference class accepts dict metadata."""
    ckpt = (Path(__file__).parent / f"checkpoints/{SIMPLE_CKPT}.yaml").absolute()
    inference = Inference(ckpt, lead_time=SIMPLE_KWARGS["lead_time"], metadata=dict_metadata)
    action = inference.from_initial_conditions(None, payload_metadata=PAYLOAD_METADATA)
    assert_payload_metadata(action, PAYLOAD_METADATA)
    assert_shape(action, {"step": 4, ENSEMBLE_DIMENSION_NAME: 1})


def test_inference_class_with_expansion_qube(dict_metadata: dict) -> None:
    """Test Inference class accepts explicit expansion_qube."""
    from qubed import Qube

    qube = Qube.from_datacube({"param": [6, 7], "level": [1, 2, 3, 4], "step": [0, 1, 2, 3]})
    inference = Inference(
        "non_existent.ckpt",
        lead_time="1D",
        metadata=dict_metadata,
        expansion_qube=qube,
    )
    _ = inference.from_initial_conditions(None)
    # The qube should be used directly without the checkpoint file
    assert inference.expansion_qube is qube


def test_inference_class_with_expansion_qube_no_metadata() -> None:
    """Test Inference class with explicit expansion_qube and no metadata."""
    from qubed import Qube

    qube = Qube.from_datacube({"param": [6, 7], "level": [1, 2, 3, 4], "step": [0, 1, 2, 3]})
    inference = Inference("non_existent.ckpt", lead_time="1D", expansion_qube=qube)
    # Should not raise even without metadata (qube is used directly)
    _ = inference.from_initial_conditions(None)
    assert inference.expansion_qube is qube


# --- utils: expansion_qube_from_metadata with dict ---


def test_expansion_qube_from_metadata_with_dict(dict_metadata: dict) -> None:
    """Test expansion_qube_from_metadata accepts dict metadata."""
    from earthkit.workflows.plugins.anemoi.utils import expansion_qube_from_metadata

    qube = expansion_qube_from_metadata(dict_metadata, "1D")
    assert "step" in qube.axes()
    assert "param" in qube.axes()
    assert "levtype" in qube.axes()


@fake_checkpoints
def test_expansion_qube_from_metadata_with_real_metadata() -> None:
    """Test expansion_qube_from_metadata still works with real Metadata objects."""
    from anemoi.inference.checkpoint import Checkpoint

    from earthkit.workflows.plugins.anemoi.utils import expansion_qube_from_metadata

    ckpt_path = Path(__file__).parent / "checkpoints" / "simple.yaml"
    metadata = Checkpoint(ckpt_path)._metadata  # type: ignore
    qube = expansion_qube_from_metadata(metadata, "1D")
    assert "step" in qube.axes()
    assert "param" in qube.axes()
    assert "levtype" in qube.axes()
