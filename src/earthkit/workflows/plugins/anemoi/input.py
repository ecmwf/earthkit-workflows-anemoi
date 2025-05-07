# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from anemoi.inference.inputs.grib import GribInput

LOG = logging.getLogger(__name__)


class ProvidedInput(GribInput):
    """
    Handles grib files
    """

    def __init__(self, context, provided_input, *, namer=None, **kwargs):
        super().__init__(context, namer=namer, **kwargs)
        self.provided_input = provided_input

    def create_input_state(self, *, date):
        return self._create_input_state(self.provided_input, variables=None, date=date)

    def load_forcings(self, *, variables, dates):
        return self._load_forcings(self.provided_input, variables=variables, dates=dates)

    def template(self, variable, date, **kwargs):
        fields = self.provided_input
        data = self._find_variable(fields, variable)
        if len(data) == 0:
            return None
        return data[0]
