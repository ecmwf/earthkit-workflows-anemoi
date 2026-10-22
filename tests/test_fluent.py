# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from anemoi.inference.testing import fake_checkpoints
from earthkit.workflows.fluent import nodetree_arrays
from earthkit.workflows.metadata import Artifacts, Requirements
from xarray import DataArray, DataTree

from earthkit.workflows.plugins.anemoi.fluent import (
    Action,
    Inference,
    from_config,
    from_initial_conditions,
    from_input,
    get_initial_conditions,
)
from earthkit.workflows.plugins.anemoi.types import ENSEMBLE_DIMENSION_NAME

STANDARD_INFERENCE_TESTS = [
    # Test inputs of ensembles
    [
        "simple",
        range(1, 3),
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 2, "param": 6, "date": 1},
    ],
    [
        "simple",
        range(1, 9),
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 8, "param": 6, "date": 1},
    ],
    [
        "simple",
        range(1, 9),
        {"date": "2020-01-01", "lead_time": "4D"},
        {"step": 16, ENSEMBLE_DIMENSION_NAME: 8, "param": 6, "date": 1},
    ],
    # Test different model configs
    [
        "simple",
        range(1, 3),
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 2, "param": 6, "date": 1},
    ],
    [
        "surface",
        range(1, 3),
        {"date": "2020-01-01", "lead_time": "1D"},
        {"step": 4, ENSEMBLE_DIMENSION_NAME: 2, "param": 6, "date": 1},
    ],
    [
        "full_atmo",
        range(1, 3),
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
        mock_config,
        ensemble_members=ensemble_members,
        **kwargs,
        checkpoint=str(ckpt_full_path),
        input="dummy",
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

    action = from_initial_conditions(ckpt_full_path, None, **kwargs)
    shape.pop("date", None)
    shape.pop(ENSEMBLE_DIMENSION_NAME, None)
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
    action = inference.from_initial_conditions(None, **kwargs)
    shape.pop("date", None)
    shape.pop(ENSEMBLE_DIMENSION_NAME, None)
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

    metadata = Checkpoint(ckpt_full_path).multi_dataset_metadata  # type: ignore

    kwargs.pop("date", None)

    action = from_initial_conditions(
        "non_existent_checkpoint.ckpt",
        None,
        metadata=metadata,
        **kwargs,
    )
    shape.pop("date", None)
    shape.pop(ENSEMBLE_DIMENSION_NAME, None)
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
        [None for _ in ensemble_members],
        dims=[ENSEMBLE_DIMENSION_NAME],
        coords={ENSEMBLE_DIMENSION_NAME: ensemble_members},
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
            [None for _ in ensemble_members],
            dims=[ENSEMBLE_DIMENSION_NAME],
            coords={ENSEMBLE_DIMENSION_NAME: ensemble_members},
        ).nodes
    )
    shape.pop("date", None)

    action = init_conditions.infer(ckpt_full_path, **kwargs)
    assert_shape(action, shape)


# --- payload_metadata propagation ---

PAYLOAD_METADATA = {"environment": ["test"], "needs_gpu": True}
SIMPLE_CKPT = "simple"
SIMPLE_KWARGS = {"date": "2020-01-01", "lead_time": "1D"}


def assert_payload_metadata(action: Action, expected: dict) -> None:
    """Assert every node in action carries the expected metadata entries."""
    for _, narray in nodetree_arrays(action.nodes):
        for node in np.atleast_1d(narray.values).flatten():
            for key, value in expected.items():
                assert (
                    getattr(node.metadata.requirements, key) == value
                ), f"Node {node.name!r} missing metadata {key!r}={value!r}, got {node.metadata.requirements}"


@fake_checkpoints
def test_from_input_propagates_payload_metadata():
    """payload_metadata passed to from_input is stored on every result node."""
    ckpt = (Path(__file__).parent / f"checkpoints/{SIMPLE_CKPT}.yaml").absolute()
    action = from_input(ckpt, "dummy", requirements=PAYLOAD_METADATA, **SIMPLE_KWARGS)
    assert_payload_metadata(action, PAYLOAD_METADATA)


@fake_checkpoints
def test_from_initial_conditions_propagates_payload_metadata():
    """payload_metadata passed to from_initial_conditions is stored on every result node."""
    ckpt = (Path(__file__).parent / f"checkpoints/{SIMPLE_CKPT}.yaml").absolute()
    kwargs = SIMPLE_KWARGS.copy()
    kwargs.pop("date")
    action = from_initial_conditions(ckpt, None, requirements=PAYLOAD_METADATA, **kwargs)
    assert_payload_metadata(action, PAYLOAD_METADATA)


@fake_checkpoints
def test_from_config_propagates_payload_metadata(mock_config):
    """payload_metadata passed to from_config is stored on every result node."""
    ckpt = (Path(__file__).parent / f"checkpoints/{SIMPLE_CKPT}.yaml").absolute()
    action = from_config(
        mock_config,
        requirements=PAYLOAD_METADATA,
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
    action = inference.from_input("dummy", SIMPLE_KWARGS["date"], requirements=PAYLOAD_METADATA)
    assert_payload_metadata(action, PAYLOAD_METADATA)


@fake_checkpoints
def test_inference_from_initial_conditions_propagates_payload_metadata():
    """payload_metadata passed to Inference.from_initial_conditions is stored on every result node."""
    ckpt = (Path(__file__).parent / f"checkpoints/{SIMPLE_CKPT}.yaml").absolute()
    inference = Inference(ckpt, lead_time=SIMPLE_KWARGS["lead_time"])
    action = inference.from_initial_conditions(None, requirements=PAYLOAD_METADATA)
    assert_payload_metadata(action, PAYLOAD_METADATA)


# --- dict metadata support ---


@pytest.fixture
def dict_metadata() -> dict:
    """Build a minimal metadata dict from the simple checkpoint YAML."""
    import yaml
    from anemoi.inference.metadata import MetadataFactory

    parent_dir = Path(__file__).parent
    with open(parent_dir / "checkpoints" / "simple.yaml") as f:
        raw = yaml.safe_load(f)
    raw_dict = {
        "config": {
            "data": raw["config"]["data"],
        },
        "data_indices": raw["data_indices"],
        "dataset": {
            "variables": raw["dataset"]["variables"],
            "variables_metadata": raw["dataset"]["variables_metadata"],
        },
    }
    metadata = MetadataFactory(raw_dict)
    return {"data": metadata}


def test_from_initial_conditions_with_dict_metadata(dict_metadata: dict) -> None:
    """Test running from initial conditions with dict metadata (no checkpoint file needed)."""
    action = from_initial_conditions(
        "non_existent.ckpt",
        None,
        metadata=dict_metadata,
        date="2020-01-01",
        lead_time="1D",
    )
    assert_shape(action, {"step": 4, "param": 6})


def test_step_coordinates_are_chronological(dict_metadata: dict) -> None:
    """Step coordinates must follow the chronological order the runner yields states in."""
    action = from_initial_conditions(
        "non_existent.ckpt",
        None,
        metadata=dict_metadata,
        date="2020-01-01",
        lead_time="2D",
    )
    step_coords = []

    def collect(node: DataTree | DataArray):
        if "step" in node.coords:
            step_coords.append(list(np.atleast_1d(node.coords["step"].values)))
        if isinstance(node, DataTree):
            for child in node.children:
                collect(node[child])

    collect(action.nodes)

    assert step_coords
    for steps in step_coords:
        assert steps == [6, 12, 18, 24, 30, 36, 42, 48]


def test_from_input_with_dict_metadata(dict_metadata: dict) -> None:
    """Test running from input with dict metadata (no checkpoint file needed)."""
    action = from_input(
        "non_existent.ckpt",
        "dummy",
        metadata=dict_metadata,
        date="2020-01-01",
        lead_time="1D",
        ensemble_members=range(1, 2),
    )
    assert_shape(action, {"step": 4, ENSEMBLE_DIMENSION_NAME: 1, "param": 6, "date": 1})


@fake_checkpoints
def test_inference_class_with_dict_metadata(dict_metadata: dict) -> None:
    """Test Inference class accepts dict metadata."""
    ckpt = (Path(__file__).parent / f"checkpoints/{SIMPLE_CKPT}.yaml").absolute()
    inference = Inference(ckpt, lead_time=SIMPLE_KWARGS["lead_time"], metadata=dict_metadata)
    action = inference.from_initial_conditions(None, requirements=PAYLOAD_METADATA)
    assert_payload_metadata(action, PAYLOAD_METADATA)
    assert_shape(action, {"step": 4})


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
    # The qube should be wrapped to {"data": qube}
    assert inference.expansion_qube["data"] is qube


def test_inference_class_with_expansion_qube_no_metadata() -> None:
    """Test Inference class with explicit expansion_qube and no metadata."""
    from qubed import Qube

    qube = Qube.from_datacube({"param": [6, 7], "level": [1, 2, 3, 4], "step": [0, 1, 2, 3]})
    inference = Inference("non_existent.ckpt", lead_time="1D", expansion_qube=qube)
    # Should not raise even without metadata (qube is used directly)
    _ = inference.from_initial_conditions(None)
    # The qube should be wrapped to {"data": qube}
    assert inference.expansion_qube["data"] is qube


# --- utils: expansion_qube_from_metadata with dict ---


def test_expansion_qube_from_metadata_with_dict(dict_metadata: dict) -> None:
    """Test expansion_qube_from_metadata accepts dict metadata."""
    from earthkit.workflows.plugins.anemoi.utils import expansion_qube_from_metadata

    qube_dict = expansion_qube_from_metadata(dict_metadata, "1D")
    qube = qube_dict["data"]
    assert "step" in qube.axes()
    assert "param" in qube.axes()
    assert "levtype" in qube.axes()


@fake_checkpoints
def test_expansion_qube_from_metadata_with_real_metadata() -> None:
    """Test expansion_qube_from_metadata still works with real Metadata objects."""
    from anemoi.inference.checkpoint import Checkpoint

    from earthkit.workflows.plugins.anemoi.utils import expansion_qube_from_metadata

    ckpt_path = Path(__file__).parent / "checkpoints" / "simple.yaml"
    qube_dict = expansion_qube_from_metadata(Checkpoint(ckpt_path).multi_dataset_metadata, "1D")
    qube = next(iter(qube_dict.values()))
    assert "step" in qube.axes()
    assert "param" in qube.axes()
    assert "levtype" in qube.axes()


# --- ensemble_members on from_initial_conditions ---

ENS_KWARGS = {"lead_time": "1D"}


@pytest.fixture
def captured_configs(monkeypatch) -> list:
    """Record the run configuration given to the model run, calling through to the real function."""
    from earthkit.workflows.plugins.anemoi import fluent as fluent_module

    configs: list = []
    original = fluent_module._run_model

    def recording(expansion_qube, config, *args, **kwargs):
        configs.append(config)
        return original(expansion_qube, config, *args, **kwargs)

    monkeypatch.setattr(fluent_module, "_run_model", recording)
    return configs


def _ens_source(members, scalar: bool = False):
    """Initial conditions action, with `number` as a dim (or a scalar coordinate)."""
    from earthkit.workflows import fluent

    if scalar:
        return fluent.from_source([None], dims=["date"], coords={ENSEMBLE_DIMENSION_NAME: members[0]})
    return fluent.from_source(
        [None for _ in members],
        dims=[ENSEMBLE_DIMENSION_NAME],
        coords={ENSEMBLE_DIMENSION_NAME: list(members)},
    )


def _ens_coords(action: Action) -> list:
    return list(action.nodes.coords[ENSEMBLE_DIMENSION_NAME].values)


def _ckpt() -> Path:
    return (Path(__file__).parent / f"checkpoints/{SIMPLE_CKPT}.yaml").absolute()


@pytest.mark.parametrize(
    "ensemble_members, expected",
    [
        [3, [3]],
        [0, [0]],
        [[3], [3]],
        [range(1, 4), [1, 2, 3]],
        [[0, 2, 5], [0, 2, 5]],
    ],
)
@fake_checkpoints
def test_inference_from_initial_conditions_exposes_members(ensemble_members, expected):
    """An action without `number` is expanded along it, single members keep a dim of length 1."""
    inference = Inference(_ckpt(), lead_time=ENS_KWARGS["lead_time"])
    action = inference.from_initial_conditions(None, ensemble_members=ensemble_members)

    assert ENSEMBLE_DIMENSION_NAME in action.nodes.dims
    assert action.nodes.sizes[ENSEMBLE_DIMENSION_NAME] == len(expected)
    assert _ens_coords(action) == expected


@fake_checkpoints
def test_inference_from_initial_conditions_members_none_unchanged():
    """None leaves an action without `number` without it."""
    inference = Inference(_ckpt(), lead_time=ENS_KWARGS["lead_time"])
    action = inference.from_initial_conditions(None)
    assert ENSEMBLE_DIMENSION_NAME not in action.nodes.dims
    assert ENSEMBLE_DIMENSION_NAME not in action.nodes.coords


@pytest.mark.parametrize("members", [[1, 2, 3], [3], [0]])
@pytest.mark.parametrize("given", [None, "same"])
@fake_checkpoints
def test_inference_from_initial_conditions_existing_number_not_exposed_again(members, given):
    """An action which already has `number` (as from `get_initial_conditions`) is not exposed twice."""
    init = _ens_source(members)
    inference = Inference(_ckpt(), lead_time=ENS_KWARGS["lead_time"])
    action = inference.from_initial_conditions(init, ensemble_members=None if given is None else members)

    assert action.nodes.sizes[ENSEMBLE_DIMENSION_NAME] == len(members)
    assert _ens_coords(action) == members
    assert_shape(action, {"step": 4, ENSEMBLE_DIMENSION_NAME: len(members), "param": 6})


@fake_checkpoints
def test_inference_from_initial_conditions_existing_number_order_insensitive():
    inference = Inference(_ckpt(), lead_time=ENS_KWARGS["lead_time"])
    action = inference.from_initial_conditions(_ens_source([1, 2, 3]), ensemble_members=[3, 1, 2])
    assert sorted(_ens_coords(action)) == [1, 2, 3]
    assert action.nodes.sizes[ENSEMBLE_DIMENSION_NAME] == 3


@pytest.mark.parametrize("scalar", [False, True])
@fake_checkpoints
def test_inference_from_initial_conditions_stacked_after_get_initial_conditions(scalar):
    """Stacking `get_initial_conditions` with `from_initial_conditions` does not double the dim."""
    from earthkit.workflows.plugins.anemoi.fluent import get_initial_conditions

    members = [3]
    if scalar:
        init = _ens_source(members, scalar=True)
        assert ENSEMBLE_DIMENSION_NAME not in init.nodes.dims
    else:
        init = get_initial_conditions(_ckpt(), "dummy", "2020-01-01", ensemble_members=3)

    inference = Inference(_ckpt(), lead_time=ENS_KWARGS["lead_time"])
    action = inference.from_initial_conditions(init, ensemble_members=3)

    # A single member is a real dimension of length 1 on the result
    assert ENSEMBLE_DIMENSION_NAME in action.nodes.dims
    assert action.nodes.sizes[ENSEMBLE_DIMENSION_NAME] == 1
    assert _ens_coords(action) == members


@pytest.mark.parametrize(
    "existing, given",
    [
        [[1, 2, 3], [1, 2]],
        [[1, 2, 3], range(1, 5)],
        [[1, 2, 3], 3],
        [[3], 4],
        [[3], [0]],
        [[3], 0],
    ],
)
@fake_checkpoints
def test_inference_from_initial_conditions_member_mismatch_raises(existing, given):
    inference = Inference(_ckpt(), lead_time=ENS_KWARGS["lead_time"])
    with pytest.raises(ValueError, match="do not match"):
        inference.from_initial_conditions(_ens_source(existing), ensemble_members=given)


@fake_checkpoints
def test_inference_from_initial_conditions_scalar_coord_mismatch_raises():
    inference = Inference(_ckpt(), lead_time=ENS_KWARGS["lead_time"])
    with pytest.raises(ValueError, match="do not match"):
        inference.from_initial_conditions(_ens_source([3], scalar=True), ensemble_members=[4])


@pytest.mark.parametrize("given", [[3], 3, [0, 1]])
@fake_checkpoints
def test_inference_from_initial_conditions_number_without_coords_raises(given):
    """Explicit members cannot be checked against a `number` dim without coordinates."""
    from earthkit.workflows import fluent

    init = fluent.from_source([None, None], dims=[ENSEMBLE_DIMENSION_NAME])
    inference = Inference(_ckpt(), lead_time=ENS_KWARGS["lead_time"])
    with pytest.raises(ValueError, match="without coordinates"):
        inference.from_initial_conditions(init, ensemble_members=given)


@fake_checkpoints
def test_inference_from_initial_conditions_number_without_coords_members_none():
    """With no explicit members a `number` dim without coordinates is left as it is."""
    from earthkit.workflows import fluent

    init = fluent.from_source([None, None], dims=[ENSEMBLE_DIMENSION_NAME])
    inference = Inference(_ckpt(), lead_time=ENS_KWARGS["lead_time"])
    action = inference.from_initial_conditions(init)
    assert action.nodes.sizes[ENSEMBLE_DIMENSION_NAME] == 2


@fake_checkpoints
def test_inference_from_initial_conditions_ensemble_members_keyword_only():
    inference = Inference(_ckpt(), lead_time=ENS_KWARGS["lead_time"])
    with pytest.raises(TypeError):
        inference.from_initial_conditions(None, None, None, [1, 2])  # type: ignore[misc]


@fake_checkpoints
def test_inference_from_initial_conditions_members_not_in_run_config(captured_configs):
    inference = Inference(_ckpt(), lead_time=ENS_KWARGS["lead_time"])
    inference.from_initial_conditions(None, ensemble_members=[1, 2])
    assert len(captured_configs) == 1
    assert "ensemble_members" not in captured_configs[0]


@fake_checkpoints
def test_module_from_initial_conditions_members_not_in_run_config(captured_configs):
    action = from_initial_conditions(_ckpt(), None, ensemble_members=[1, 2], **ENS_KWARGS)
    assert len(captured_configs) == 1
    assert "ensemble_members" not in captured_configs[0]
    assert _ens_coords(action) == [1, 2]


@fake_checkpoints
def test_from_initial_conditions_none_placeholder_labelling_tasks_execute():
    """The labelling tasks of a `None` initial conditions placeholder run, and pass `None` through.

    The member ids stay on the `number` dimension of the action.
    """
    from cascade.low.core import TaskDefinition

    from earthkit.workflows.plugins.anemoi.utils import expose_ensemble_dimension

    action = from_initial_conditions(_ckpt(), None, ensemble_members=[0, 3], **ENS_KWARGS)
    assert _ens_coords(action) == [0, 3]

    labelling = []
    for node in action.graph().nodes():
        definition = node.payload.definition
        if definition.func is None:
            continue
        func = TaskDefinition.func_dec(definition.func)
        if func is expose_ensemble_dimension:
            labelling.append(node.payload)

    # Upstream value of the initial conditions placeholder is `None`, the member is a static argument
    assert sorted(payload.static_input_ps["1"] for payload in labelling) == [0, 3]
    for payload in labelling:
        func = TaskDefinition.func_dec(payload.definition.func)
        assert func(None, payload.static_input_ps["1"]) is None


@fake_checkpoints
def test_action_infer_members_not_in_run_config(captured_configs):
    action = _ens_source([1, 2, 3])
    result = Action(action.nodes).infer(_ckpt(), ensemble_members=[1, 2, 3], **ENS_KWARGS)
    assert len(captured_configs) == 1
    assert "ensemble_members" not in captured_configs[0]
    assert _ens_coords(result) == [1, 2, 3]

    # Without members the number of members is inferred from the action
    result = Action(action.nodes).infer(_ckpt(), **ENS_KWARGS)
    assert "ensemble_members" not in captured_configs[1]
    assert _ens_coords(result) == [1, 2, 3]


# --- per-stage requirements, artifacts and the deprecated environment ---

SELF_PACKAGE = "earthkit-workflows-anemoi"
ARTIFACTS = Artifacts(artifact_urls={"model": "https://example.com/model"})


def simple_ckpt() -> Path:
    return (Path(__file__).parent / f"checkpoints/{SIMPLE_CKPT}.yaml").absolute()


def graph_nodes(action: Action) -> list:
    """All nodes the result of an action depends on, including the result nodes."""
    seen = {}
    stack = []
    for _, narray in nodetree_arrays(action.nodes):
        stack.extend(np.atleast_1d(narray.values).flatten())
    while stack:
        node = stack.pop()
        if node.name in seen:
            continue
        seen[node.name] = node
        stack.extend(output.parent for output in node.inputs.values())
    return list(seen.values())


def stage_nodes(action: Action) -> tuple[list, list, list]:
    """Split the graph into model nodes, initial condition source nodes and the other (expansion) nodes."""
    nodes = graph_nodes(action)
    model = [n for n in nodes if n.payload.definition.entrypoint.endswith("run_as_earthkit")]
    sources = [n for n in nodes if not n.inputs]
    others = [n for n in nodes if n not in model and n not in sources]
    assert model and sources and others
    return model, sources, others


def environment_of(node) -> set[str]:
    return set(node.metadata.requirements.environment)


def deprecation_warnings(records: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [r for r in records if issubclass(r.category, DeprecationWarning) and "environment" in str(r.message)]


@fake_checkpoints
def test_per_stage_requirements_reach_the_right_nodes():
    """A dict keyed by stage puts each requirements on the nodes of that stage only."""
    inference = Inference(simple_ckpt(), lead_time="1D")
    action = inference.from_input(
        "dummy",
        "2020-01-01",
        requirements={
            "inference": Requirements(environment=["model-pkg"]),
            "initial_conditions": {"environment": ["ic-pkg"]},
        },
    )
    model, sources, others = stage_nodes(action)
    assert all(environment_of(n) == {"model-pkg"} for n in model)
    assert all(environment_of(n) == {"ic-pkg"} for n in sources)
    assert all(n.metadata.requirements.needs_gpu is None for n in sources)
    assert all(environment_of(n) == {"model-pkg"} for n in others)


@fake_checkpoints
def test_per_stage_requirements_with_missing_stage():
    """A stage missing from the dict gets no requirements."""
    action = Inference(simple_ckpt(), lead_time="1D").from_input(
        "dummy", "2020-01-01", requirements={"initial_conditions": {"environment": ["ic-pkg"]}}
    )
    model, sources, _ = stage_nodes(action)
    assert all(environment_of(n) == set() for n in model)
    assert all(environment_of(n) == {"ic-pkg"} for n in sources)


@fake_checkpoints
def test_per_stage_requirements_from_config(mock_config):
    """Per-stage requirements reach the right nodes in `from_config`."""
    action = from_config(
        mock_config,
        requirements={"inference": {"environment": ["model-pkg"]}, "initial_conditions": {"environment": ["ic-pkg"]}},
        checkpoint=str(simple_ckpt()),
        input="dummy",
        **SIMPLE_KWARGS,
    )
    model, sources, _ = stage_nodes(action)
    assert all(environment_of(n) == {"model-pkg"} for n in model)
    assert all(environment_of(n) == {"ic-pkg"} for n in sources)


@fake_checkpoints
def test_single_requirements_reach_initial_conditions_and_inference():
    """A single `Requirements` applies to the initial condition and the inference nodes."""
    requirements = Requirements(environment=["shared"], needs_gpu=True)
    action = Inference(simple_ckpt(), lead_time="1D").from_input("dummy", "2020-01-01", requirements=requirements)
    model, sources, others = stage_nodes(action)
    for node in [*model, *sources, *others]:
        assert environment_of(node) == {"shared"}
        assert node.metadata.requirements.needs_gpu is True


@fake_checkpoints
def test_single_requirements_reach_from_initial_conditions_source():
    """The source nodes built by `from_initial_conditions` get the requirements too."""
    action = Inference(simple_ckpt(), lead_time="1D").from_initial_conditions(
        None, requirements=Requirements(environment=["shared"])
    )
    model, sources, _ = stage_nodes(action)
    assert all(environment_of(n) == {"shared"} for n in [*model, *sources])


@fake_checkpoints
def test_ensemble_expansion_gets_initial_conditions_stage():
    """The ensemble member expansion of `from_initial_conditions` is part of the `initial_conditions` stage."""
    action = Inference(simple_ckpt(), lead_time="1D").from_initial_conditions(
        None,
        ensemble_members=[1, 2],
        requirements={"inference": {"environment": ["model-pkg"]}, "initial_conditions": {"environment": ["ic-pkg"]}},
        artifacts=ARTIFACTS,
    )
    model, sources, _ = stage_nodes(action)
    # Nodes between the source and the model nodes are the ensemble member expansion
    expansion = {}
    stack = [output.parent for node in model for output in node.inputs.values()]
    while stack:
        node = stack.pop()
        if node in sources or node.name in expansion:
            continue
        expansion[node.name] = node
        stack.extend(output.parent for output in node.inputs.values())
    assert expansion
    for node in [*expansion.values(), *sources]:
        assert environment_of(node) == {"ic-pkg"}
        assert node.metadata.artifacts.artifact_urls == ARTIFACTS.artifact_urls


@fake_checkpoints
def test_from_initial_conditions_requirements_are_positional():
    """`requirements` is still the second positional argument of `Inference.from_initial_conditions`."""
    action = Inference(simple_ckpt(), lead_time="1D").from_initial_conditions(
        None, {"inference": {"environment": ["model-pkg"]}, "initial_conditions": {"environment": ["ic-pkg"]}}
    )
    model, sources, _ = stage_nodes(action)
    assert all(environment_of(n) == {"model-pkg"} for n in model)
    assert all(environment_of(n) == {"ic-pkg"} for n in sources)


@fake_checkpoints
def test_model_nodes_keep_needs_gpu():
    """Model nodes need a GPU, with or without requirements, and nothing else does by default."""
    for requirements in (None, Requirements(environment=["x"]), {"inference": {"environment": ["x"]}}):
        action = Inference(simple_ckpt(), lead_time="1D").from_input("dummy", "2020-01-01", requirements=requirements)
        model, sources, _ = stage_nodes(action)
        assert all(n.metadata.requirements.needs_gpu is True for n in model)
        assert all(n.metadata.requirements.needs_gpu is None for n in sources)


@fake_checkpoints
def test_artifacts_reach_initial_conditions_and_inference(mock_config):
    """Artifacts are on every node created by a call, initial conditions included."""
    ckpt = simple_ckpt()
    actions = [
        Inference(ckpt, lead_time="1D").from_input("dummy", "2020-01-01", artifacts=ARTIFACTS),
        Inference(ckpt, lead_time="1D").from_initial_conditions(None, artifacts=ARTIFACTS),
        from_input(ckpt, "dummy", artifacts=ARTIFACTS, **SIMPLE_KWARGS),
        from_initial_conditions(ckpt, None, artifacts=ARTIFACTS, lead_time="1D"),
        from_config(mock_config, artifacts=ARTIFACTS, checkpoint=str(ckpt), input="dummy", **SIMPLE_KWARGS),
    ]
    for action in actions:
        model, sources, others = stage_nodes(action)
        for node in [*model, *sources, *others]:
            assert node.metadata.artifacts.artifact_urls == ARTIFACTS.artifact_urls


@fake_checkpoints
def test_user_objects_are_not_modified_and_can_be_reused():
    """One `Requirements` and one `Artifacts` used in several calls are unchanged afterwards."""
    requirements = Requirements(environment=["shared"], needs_gpu=True)
    per_stage = {"inference": Requirements(environment=["model-pkg"]), "initial_conditions": {"environment": ["ic"]}}
    artifacts = Artifacts(artifact_urls={"a": "b"})
    requirements_before = requirements.model_copy(deep=True)
    per_stage_before = {
        "inference": per_stage["inference"].model_copy(deep=True),
        "initial_conditions": {"environment": ["ic"]},
    }
    artifacts_before = artifacts.model_copy(deep=True)

    inference = Inference(simple_ckpt(), lead_time="1D", requirements=requirements)
    for _ in range(2):
        inference.from_input("dummy", "2020-01-01", requirements=requirements, artifacts=artifacts)
        inference.from_initial_conditions(None, requirements, artifacts)
        inference.from_input("dummy", "2020-01-01", requirements=per_stage, artifacts=artifacts)
        from_input(simple_ckpt(), "dummy", requirements=requirements, artifacts=artifacts, **SIMPLE_KWARGS)

    assert requirements == requirements_before
    assert per_stage == per_stage_before
    assert artifacts == artifacts_before


@fake_checkpoints
def test_constructor_and_method_requirements_are_merged():
    """Requirements of the constructor and of the method are merged for each stage."""
    inference = Inference(
        simple_ckpt(),
        lead_time="1D",
        requirements={
            "inference": {"environment": ["from-init"], "needs_gpu": True},
            "initial_conditions": {"environment": ["ic-init"]},
        },
    )
    action = inference.from_input(
        "dummy", "2020-01-01", requirements={"inference": Requirements(environment=["from-call"])}
    )
    model, sources, _ = stage_nodes(action)
    assert all(environment_of(n) == {"from-init", "from-call"} for n in model)
    assert all(environment_of(n) == {"ic-init"} for n in sources)

    action = inference.from_initial_conditions(None, Requirements(environment=["everywhere"]))
    model, sources, _ = stage_nodes(action)
    assert all(environment_of(n) == {"from-init", "everywhere"} for n in model)
    assert all(environment_of(n) == {"ic-init", "everywhere"} for n in sources)


@fake_checkpoints
def test_constructor_requirements_alone_apply():
    """Requirements given only to the constructor apply to later calls."""
    inference = Inference(simple_ckpt(), lead_time="1D", requirements=Requirements(environment=["init-only"]))
    model, sources, _ = stage_nodes(inference.from_input("dummy", "2020-01-01"))
    assert all(environment_of(n) == {"init-only"} for n in [*model, *sources])


def existing_initial_conditions() -> Action:
    from earthkit.workflows import fluent

    return Action(
        fluent.from_source([None], dims=[ENSEMBLE_DIMENSION_NAME], coords={ENSEMBLE_DIMENSION_NAME: [0]}).nodes
    )


DEPRECATED_CALLS = {
    "Inference": lambda ckpt, config, env: Inference(ckpt, lead_time="1D", environment=env),
    "from_input": lambda ckpt, config, env: from_input(ckpt, "dummy", environment=env, **SIMPLE_KWARGS),
    "from_initial_conditions": lambda ckpt, config, env: from_initial_conditions(
        ckpt, None, lead_time="1D", environment=env
    ),
    "from_config": lambda ckpt, config, env: from_config(
        config, environment=env, checkpoint=str(ckpt), input="dummy", **SIMPLE_KWARGS
    ),
    "Action.infer": lambda ckpt, config, env: existing_initial_conditions().infer(ckpt, "1D", environment=env),
}


@pytest.mark.parametrize("name", list(DEPRECATED_CALLS))
@fake_checkpoints
def test_deprecated_environment_warns_once_and_applies(name, mock_config):
    """`environment` warns exactly once, at the line of the caller, and still reaches the nodes."""
    environment = ["legacy-pkg"]
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = DEPRECATED_CALLS[name](simple_ckpt(), mock_config, environment)

    found = deprecation_warnings(records)
    assert len(found) == 1, [str(r.message) for r in found]
    assert found[0].filename == __file__
    assert environment == ["legacy-pkg"]

    if name == "Inference":
        result = result.from_initial_conditions(None)
    model, sources, _ = stage_nodes(result)
    assert all(environment_of(n) == {"legacy-pkg"} for n in model)
    if name != "Action.infer":  # the graph given to `infer` already exists
        assert all(environment_of(n) == {"legacy-pkg"} for n in sources)


@fake_checkpoints
def test_no_warning_without_environment(mock_config):
    """Nothing is warned about when `environment` is not used."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        for name, call in DEPRECATED_CALLS.items():
            call(simple_ckpt(), mock_config, None)
    assert deprecation_warnings(records) == []


@fake_checkpoints
def test_deprecated_environment_and_requirements_are_merged():
    """`environment` and `requirements` are merged, not one over the other."""
    with pytest.warns(DeprecationWarning, match="environment"):
        action = from_input(
            simple_ckpt(),
            "dummy",
            environment=["legacy-pkg"],
            requirements=Requirements(environment=["new-pkg"], needs_gpu=True),
            **SIMPLE_KWARGS,
        )
    model, sources, others = stage_nodes(action)
    for node in [*model, *sources, *others]:
        assert environment_of(node) == {"legacy-pkg", "new-pkg"}
        assert node.metadata.requirements.needs_gpu is True


@fake_checkpoints
def test_deprecated_environment_dict_is_per_stage():
    """A dict `environment` is keyed by stage."""
    with pytest.warns(DeprecationWarning, match="environment"):
        action = from_input(
            simple_ckpt(),
            "dummy",
            environment={"inference": ["model-pkg"], "initial_conditions": ["ic-pkg"]},
            **SIMPLE_KWARGS,
        )
    model, sources, _ = stage_nodes(action)
    assert all(environment_of(n) == {"model-pkg"} for n in model)
    assert all(environment_of(n) == {"ic-pkg"} for n in sources)


def test_fiab_shaped_calls(monkeypatch):
    """The way forecast-in-a-box calls the class keeps working, with one warning and one self pin."""
    from qubed import Qube

    monkeypatch.setattr("earthkit.workflows.plugins.anemoi.__version__", "1.2.3")
    own_pin = f"{SELF_PACKAGE}==9.9.9"
    environment = ["some-pkg==1.0", own_pin]
    qube = Qube.from_datacube({"param": [6, 7], "level": [1, 2, 3, 4], "step": [0, 1, 2, 3]})
    artifacts = Artifacts(artifact_urls={"model-id": "https://example.com/model"})

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        inference = Inference(
            ckpt="non_existent.ckpt",
            lead_time="1D",
            environment=environment,
            expansion_qube=qube,
        )
        from_input_action = inference.from_input(
            input="dummy", date="2020-01-01", ensemble_members=2, artifacts=artifacts
        )
        from_ic_action = inference.from_initial_conditions(None, artifacts=artifacts)

    found = deprecation_warnings(records)
    assert len(found) == 1
    assert found[0].filename == __file__
    assert environment == ["some-pkg==1.0", own_pin]

    for action in (from_input_action, from_ic_action):
        model, sources, others = stage_nodes(action)
        for node in [*model, *sources, *others]:
            assert node.metadata.artifacts.artifact_urls == artifacts.artifact_urls
        for node in model:
            assert environment_of(node) == set(environment)
            assert [e for e in node.metadata.requirements.environment if e.startswith(SELF_PACKAGE)] == [own_pin]
            assert node.metadata.requirements.needs_gpu is True


def test_self_pin_is_not_added_twice_when_merging(monkeypatch):
    """An own pin from the constructor is not joined by an automatic one from the method requirements."""
    from qubed import Qube

    monkeypatch.setattr("earthkit.workflows.plugins.anemoi.__version__", "1.2.3")
    qube = Qube.from_datacube({"param": [6, 7], "level": [1, 2, 3, 4], "step": [0, 1, 2, 3]})
    inference = Inference(
        "non_existent.ckpt",
        lead_time="1D",
        expansion_qube=qube,
        requirements=Requirements(environment=[f"{SELF_PACKAGE}==9.9.9"]),
    )
    action = inference.from_initial_conditions(None, Requirements(environment=["extra"]))
    model, sources, _ = stage_nodes(action)
    for node in [*model, *sources]:
        assert environment_of(node) == {f"{SELF_PACKAGE}==9.9.9", "extra"}


def test_self_pin_is_added_per_stage(monkeypatch):
    """The package itself is added to every stage that has an environment."""
    from qubed import Qube

    monkeypatch.setattr("earthkit.workflows.plugins.anemoi.__version__", "1.2.3")
    qube = Qube.from_datacube({"param": [6, 7], "level": [1, 2, 3, 4], "step": [0, 1, 2, 3]})
    action = Inference("non_existent.ckpt", lead_time="1D", expansion_qube=qube).from_initial_conditions(
        None, {"inference": {"environment": ["x"]}, "initial_conditions": {"environment": ["y"]}}
    )
    model, sources, _ = stage_nodes(action)
    assert all(environment_of(n) == {"x", f"{SELF_PACKAGE}~=1.2.3"} for n in model)
    assert all(environment_of(n) == {"y", f"{SELF_PACKAGE}~=1.2.3"} for n in sources)


# --- get_initial_conditions ---


def test_get_initial_conditions_stage_requirements_and_artifacts():
    """Only the `initial_conditions` stage is used, and artifacts reach every node."""
    action = get_initial_conditions(
        simple_ckpt(),
        "dummy",
        "2020-01-01",
        ensemble_members=[1, 2],
        requirements={
            "inference": {"environment": ["model-pkg"], "needs_gpu": True},
            "initial_conditions": Requirements(environment=["ic-pkg"]),
        },
        artifacts=ARTIFACTS,
    )
    nodes = graph_nodes(action)
    assert len(nodes) == 2
    for node in nodes:
        assert environment_of(node) == {"ic-pkg"}
        assert node.metadata.requirements.needs_gpu is None
        assert node.metadata.artifacts.artifact_urls == ARTIFACTS.artifact_urls


def test_get_initial_conditions_single_requirements():
    action = get_initial_conditions(
        simple_ckpt(), "dummy", "2020-01-01", requirements=Requirements(environment=["shared"], needs_gpu=True)
    )
    for node in graph_nodes(action):
        assert environment_of(node) == {"shared"}
        assert node.metadata.requirements.needs_gpu is True


@pytest.mark.parametrize(
    "environment", [["legacy-pkg"], {"initial_conditions": ["legacy-pkg"], "inference": ["other"]}]
)
def test_get_initial_conditions_deprecated_environment(environment):
    """`environment` warns once at the caller's line, and is merged with `requirements`."""
    requirements = Requirements(environment=["new-pkg"], needs_gpu=True)
    before = requirements.model_copy(deep=True)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        action = get_initial_conditions(
            simple_ckpt(), "dummy", "2020-01-01", environment=environment, requirements=requirements
        )

    found = deprecation_warnings(records)
    assert len(found) == 1, [str(r.message) for r in found]
    assert found[0].filename == __file__
    assert requirements == before
    for node in graph_nodes(action):
        assert environment_of(node) == {"legacy-pkg", "new-pkg"}
        assert node.metadata.requirements.needs_gpu is True


def test_get_initial_conditions_no_warning_without_environment():
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        get_initial_conditions(simple_ckpt(), "dummy", "2020-01-01", requirements=Requirements(environment=["x"]))
    assert deprecation_warnings(records) == []
