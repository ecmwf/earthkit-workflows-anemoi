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
