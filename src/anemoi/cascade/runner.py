
"""
Custom Cascade Runner

Used for when providing initial conditions
"""

from __future__ import annotations

import logging
from typing import Any


from anemoi.utils.config import DotDict
from pydantic import BaseModel

from anemoi.inference.forcings import BoundaryForcings, ComputedForcings, CoupledForcings
from anemoi.inference.runner import Runner
from anemoi.inference.inputs import create_input
from anemoi.inference.config import Configuration

LOG = logging.getLogger(__name__)

class CascadeRunner(Runner):
    def __init__(self, config: dict | Configuration, **kwargs):

        if isinstance(config, dict):
            # So we get the dot notation
            config = DotDict(config)

        # Remove that when the Pydantic model is ready
        if isinstance(config, BaseModel):
            config = DotDict(config.model_dump())

        self.config = config

        default_init_args = dict(
            checkpoint = config.checkpoint,
            device=config.device,
            precision=config.precision,
            allow_nans=config.allow_nans,
            verbosity=config.verbosity,
            report_error=config.report_error,
            use_grib_paramid=config.use_grib_paramid,
            development_hacks=config.development_hacks,
        )
        default_init_args.update(kwargs)

        super().__init__(
            **default_init_args
        )

    def create_input(self):
        input = create_input(self, self.config.input)
        LOG.info("Input: %s", input)
        return input

    # Computed forcings
    def create_constant_computed_forcings(self, variables, mask):
        result = ComputedForcings(self, variables, mask)
        LOG.info("Constant computed forcing: %s", result)
        return result

    def create_dynamic_computed_forcings(self, variables, mask):
        result = ComputedForcings(self, variables, mask)
        LOG.info("Dynamic computed forcing: %s", result)
        return result

    def _input_forcings(self, name):
        if self.config.forcings is None:
            # Use the same as the input
            return self.config.input

        if name in self.config.forcings:
            return self.config.forcings[name]

        if "input" in self.config.forcings:
            return self.config.forcings.input

        return self.config.forcings

    def create_constant_coupled_forcings(self, variables, mask):
        input = create_input(self, self._input_forcings("constant"))
        result = CoupledForcings(self, input, variables, mask)
        LOG.info("Constant coupled forcing: %s", result)
        return result

    def create_dynamic_coupled_forcings(self, variables, mask):

        input = create_input(self, self._input_forcings("dynamic"))
        result = CoupledForcings(self, input, variables, mask)
        LOG.info("Dynamic coupled forcing: %s", result)
        return result

    def create_boundary_forcings(self, variables, mask):

        input = create_input(self, self._input_forcings("boundary"))
        result = BoundaryForcings(self, input, variables, mask)
        LOG.info("Boundary forcing: %s", result)
        return result


    @staticmethod
    def from_kwargs(checkpoint: str | dict[str, Any], configuration_kwargs: dict[str, Any] | None = None, **kwargs) -> 'CascadeRunner':
        config = Configuration(checkpoint = checkpoint, **(configuration_kwargs or {}))
        return CascadeRunner(config, **kwargs)
