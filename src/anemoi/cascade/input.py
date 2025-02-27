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
