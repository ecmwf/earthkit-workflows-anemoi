# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from anemoi.inference.testing import fake_checkpoints

from earthkit.workflows.plugins.anemoi.inference import _get_initial_conditions_from_config


@fake_checkpoints
def test_get_initial_conditions(mock_config):
    """Test getting initial conditions"""
    init_conditions = _get_initial_conditions_from_config(mock_config, date="2000-01-01")
    assert init_conditions is not None
