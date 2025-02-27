import pytest

from anemoi.cascade.fluent import from_initial_conditions
from anemoi.cascade.fluent import get_initial_conditions_source

from .mocks import MockInput
from .mocks import mocked_metadata

_ = mocked_metadata


@pytest.fixture
def mock_input():
    return MockInput()


@pytest.mark.parametrize(
    "date, ensemble_members, perturbation, shape",
    [
        ["2000-01-01", 1, False, {"date": 1}],
        ["2000-01-01", [2], False, {"date": 1}],
        ["2000-01-01", range(1, 2), False, {"date": 1}],
        ["2000-01-01", 10, False, {"date": 1, "ensemble_member": 10}],
        ["2000-01-01", range(10), False, {"date": 1, "ensemble_member": 10}],
        ["2000-01-01", range(10, 20), False, {"date": 1, "ensemble_member": 10}],
        ["2000-01-01", 10, True, {"date": 1, "ensemble_member": 10}],
        ["2000-01-01", 51, False, {"date": 1, "ensemble_member": 51}],
        ["2000-01-01", 51, True, {"date": 1, "ensemble_member": 51}],
        [(2000, 1, 1), 51, False, {"date": 1, "ensemble_member": 51}],
        [(2000, 1, 1), 51, True, {"date": 1, "ensemble_member": 51}],
    ],
)
def test_get_initial_conditions(mock_input, date, ensemble_members, perturbation, shape):
    """Test getting initial conditions"""
    action = get_initial_conditions_source(
        mock_input, date, ensemble_members, initial_condition_perturbation=perturbation
    )

    for dim in shape:
        assert dim in action.nodes.dims
        assert action.nodes.coords[dim].size == shape[dim]


@pytest.mark.parametrize(
    "date, ensemble_members, perturbation, shape",
    [
        ["2000-01-01", 0, False, {"date": 1, "ensemble_member": 10}],
        ["2000-01-01", -1, False, {"date": 1, "ensemble_member": 10}],
    ],
)
def test_get_initial_conditions_fail(mock_input, date, ensemble_members, perturbation, shape):
    """Test failing to get initial conditions"""
    with pytest.raises(ValueError):
        _ = get_initial_conditions_source(
            mock_input, date, ensemble_members, initial_condition_perturbation=perturbation
        )


@pytest.fixture
def mocked_initial_conditions(mock_input):
    return get_initial_conditions_source(mock_input, "2000-01-01", 1)


@pytest.fixture
def mocked_ten_initial_conditions(mock_input):
    return get_initial_conditions_source(mock_input, "2000-01-01", 10)


@pytest.fixture
def mocked_checkpoint_file(tmp_path) -> str:
    tmp_ckpt = tmp_path / "mocked_checkpoint.ckpt"
    tmp_ckpt.touch()
    return str(tmp_ckpt)


@pytest.fixture
def anemoi_config(mocked_checkpoint_file):
    """Mock a configuration file with a real checkpoint file"""
    return {
        "checkpoint": str(mocked_checkpoint_file),
    }


@pytest.fixture
def mocked_config_file(tmp_path, anemoi_config):
    """Mock a written configuration file"""
    config_file = tmp_path / "config.yaml"
    import yaml

    yaml.safe_dump(anemoi_config, config_file.open("w"))

    return config_file


@pytest.fixture
def mocked_load_metadata(mocker, mocked_metadata, **kwargs):
    metadata = mocked_metadata
    metadata.update(kwargs)

    mocker.patch("anemoi.inference.checkpoint.load_metadata", return_value=metadata)


@pytest.mark.parametrize(
    "ensemble_members, kwargs, shape",
    [
        [2, {"lead_time": "1D"}, {"step": 4, "ensemble_member": 2, "param": 30}],
        [8, {"lead_time": "1D"}, {"step": 4, "ensemble_member": 8, "param": 30}],
        [8, {"lead_time": "4D"}, {"step": 16, "ensemble_member": 8, "param": 30}],
    ],
)
def test_from_initial_conditions(
    mocked_checkpoint_file, mocked_initial_conditions, mocked_load_metadata, ensemble_members, kwargs, shape
):
    """Test running from initial conditions"""
    _ = mocked_load_metadata
    action = from_initial_conditions(
        mocked_checkpoint_file, mocked_initial_conditions, ensemble_members=ensemble_members, **kwargs
    )

    for dim in shape:
        assert dim in action.nodes.dims
        assert action.nodes.coords[dim].size == shape[dim]


@pytest.mark.parametrize(
    "ensemble_members, kwargs, shape",
    [
        [10, {"lead_time": "1D"}, {"step": 4, "ensemble_member": 10, "param": 30}],
        [10, {"lead_time": "4D"}, {"step": 16, "ensemble_member": 10, "param": 30}],
        [None, {"lead_time": "4D"}, {"step": 16, "ensemble_member": 10, "param": 30}],
    ],
)
def test_from_many_initial_conditions(
    mocked_checkpoint_file, mocked_ten_initial_conditions, mocked_load_metadata, ensemble_members, kwargs, shape
):
    """Test running from many initial conditions"""
    _ = mocked_load_metadata
    action = from_initial_conditions(
        mocked_checkpoint_file, mocked_ten_initial_conditions, ensemble_members=ensemble_members, **kwargs
    )

    for dim in shape:
        assert dim in action.nodes.dims
        assert action.nodes.coords[dim].size == shape[dim]
