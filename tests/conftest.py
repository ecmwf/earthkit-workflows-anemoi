# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from .executor import spawn_gateway

pytest_plugins = "anemoi.utils.testing"


@pytest.fixture(scope="session")
def shared_gateway():
    """Shared gateway process and URL for all tests to avoid ZMQ conflicts."""
    url, process = spawn_gateway(max_jobs=4)
    yield url
    if process and process.is_alive():
        process.kill()
