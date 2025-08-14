# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Custom Cascade Runner

Used for when providing initial conditions
"""

from __future__ import annotations

import logging
from typing import List

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.forcings import ComputedForcings
from anemoi.inference.forcings import CoupledForcings
from anemoi.inference.forcings import Forcings
from anemoi.inference.inputs import create_input
from anemoi.inference.inputs.ekd import EkdInput
from anemoi.inference.post_processors import create_post_processor
from anemoi.inference.processor import Processor
from anemoi.inference.runner import Runner
from anemoi.inference.types import IntArray

LOG = logging.getLogger(__name__)


class CascadeRunner(Runner):
    """Cascade Inference Runner"""

    def __init__(self, config: RunConfiguration | dict):
        if isinstance(config, dict):
            config = RunConfiguration(**config)

        self.config = config

        super().__init__(
            config.checkpoint,  # type: ignore # Error in anemoi.inference
            device=config.device,
            precision=config.precision,
            allow_nans=config.allow_nans,
            verbosity=config.verbosity,
            report_error=config.report_error,
            patch_metadata=config.patch_metadata,
            typed_variables=config.typed_variables,
        )

    def create_input(self) -> EkdInput:
        """Create the input.

        Returns
        -------
        Input
            The created input.
        """
        input = create_input(self, self.config.input)

        def set_variables(input: EkdInput) -> EkdInput:
            """Set the variables for the input."""
            if not isinstance(input, EkdInput):
                LOG.warning("Input is not an instance of EkdInput, setting the expected variables may not work.")
            input.variables = input.checkpoint.variables_from_input(include_forcings=True)
            return input

        from anemoi.inference.inputs.cutout import Cutout

        if isinstance(input, Cutout):  # type: ignore
            input.sources = {src: set_variables(src_input) for src, src_input in input.sources.items()}
        elif isinstance(input, EkdInput):
            input = set_variables(input)
        # If input is not an instance of EkdInput, we cannot set the variables
        # but we can still use the input as it is.
        # This is a fallback for when the input is not an instance of EkdInput.
        LOG.info("Input: %s", input)
        return input

    def create_constant_computed_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        """Create constant computed forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List[Forcings]
            The created constant computed forcings.
        """
        result = ComputedForcings(self, variables, mask)
        LOG.info("Constant computed forcing: %s", result)
        return [result]

    def create_dynamic_computed_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        """Create dynamic computed forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List[Forcings]
            The created dynamic computed forcings.
        """
        result = ComputedForcings(self, variables, mask)
        LOG.info("Dynamic computed forcing: %s", result)
        return [result]

    def create_constant_coupled_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        """Create constant coupled forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List
            The created constant coupled forcings.
        """
        # This runner does not support coupled forcings
        # there are supposed to be already in the state dictionary
        # or managed by the user.
        input = create_input(self, self.config.input)
        # return []
        result = CoupledForcings(self, input, variables, mask)
        LOG.info("Constant coupled forcing: %s", result)
        return [result]

    def create_dynamic_coupled_forcings(self, variables: List[str], mask: IntArray) -> List[Forcings]:
        """Create dynamic coupled forcings.

        Parameters
        ----------
        variables : List[str]
            The variables for the forcings.
        mask : IntArray
            The mask for the forcings.

        Returns
        -------
        List
            The created dynamic coupled forcings.
        """
        # This runner does not support coupled forcings
        # there are supposed to be already in the state dictionary
        # or managed by the user.
        LOG.warning("Coupled forcings are not supported by this runner: %s", variables)
        return []

    def create_post_processors(self) -> List[Processor]:
        """Create post-processors.

        Returns
        -------
        List[Processor]
            The created post-processors.
        """
        result = []
        for processor in self.config.post_processors:
            result.append(create_post_processor(self, processor))

        LOG.info("Post processors: %s", result)
        return result
