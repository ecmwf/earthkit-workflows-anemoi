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
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from datetime import datetime
from io import BytesIO

import earthkit.data as ekd
from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.grib.templates import template_provider_registry
from anemoi.inference.metadata import Metadata
from anemoi.inference.output import Output
from anemoi.inference.outputs import output_registry
from anemoi.inference.outputs.gribmemory import GribMemoryOutput
from anemoi.inference.runner import Runner
from anemoi.inference.types import State
from anemoi.utils.grib import shortname_to_paramid

from .types import ENSEMBLE_DIMENSION_NAME

LOG = logging.getLogger(__name__)


@output_registry.register("cascade")
class CascadeOutput(Output):
    """Custom output class for the CascadeRunner that converts model states to GRIB format and then to earthkit FieldList."""

    def __init__(self, runner: CascadeRunner, metadata: Metadata):
        super().__init__(runner, metadata)

        mir_templates_available = template_provider_registry.is_registered("mir")
        self._templates = (["mir"] if mir_templates_available else []) + ["builtin"]

    def write_step(self, state: dict) -> ekd.SimpleFieldList:  # type: ignore[reportIncompatibleMethodOverride]
        initial_date: datetime = state["date"]
        ensemble_member = state.get(ENSEMBLE_DIMENSION_NAME, None)
        grib_metadata = {}

        grib_metadata.update(
            {
                "edition": 2,
                "type": "fc",
                "class": "ai",
            }
        )
        if ensemble_member is not None:
            grib_metadata.update(
                {
                    "productDefinitionTemplateNumber": 1,
                    "type": "pf",
                    "stream": "enfo",
                    "number": ensemble_member,
                }
            )

        try:
            grib_memory = BytesIO()
            grib_output = GribMemoryOutput(
                self.context, self.metadata, out=grib_memory, templates=self._templates, encoding=grib_metadata, check_encoding = False
            )
            grib_output.write_state(state)
            grib_memory.seek(0, 0)
            fieldlist: ekd.SimpleFieldList = ekd.from_source("stream", grib_memory, read_all=True)  # type: ignore[reportAssignmentType]

            return fieldlist

        except Exception:
            LOG.error("Error converting state to grib, will convert to ArrayField.", exc_info=True)

        import numpy as np

        fields = []

        step = state["step"]

        for var, array in state["fields"].items():
            variable = self.typed_variables[var]
            paramId = shortname_to_paramid(variable.param)

            grib_metadata.update(
                {
                    "step": step,
                    "base_datetime": initial_date,
                    "paramId": paramId,
                    "shortName": variable.param,
                    "param": variable.param,
                    "latitudes": state["latitudes"],
                    "longitudes": np.where(
                        state["longitudes"] > 180,
                        state["longitudes"] - 360,
                        state["longitudes"],
                    ),
                }
            )
            if "levtype" in variable.grib_keys:
                grib_metadata["levtype"] = variable.grib_keys["levtype"]
            if variable.level is not None:
                grib_metadata["level"] = variable.level

            fields.append(ekd.ArrayField(array, grib_metadata.copy()))

        return ekd.SimpleFieldList.from_fields(fields)


class CascadeRunner(Runner):
    """Cascade Inference Runner"""

    outputs: dict[str, CascadeOutput]  # type: ignore[reportIncompatibleVariableOverride]

    def __init__(self, checkpoint: str, **kwargs):
        """Initialise the CascadeRunner.

        Parameters
        ----------
        checkpoint : str
            Path to the model checkpoint.
        **kwargs : Any
            Additional keyword arguments passed to the `RunConfiguration`.
        """
        config = RunConfiguration(
            checkpoint=checkpoint,
            output="cascade",
            **kwargs,
        )
        super().__init__(config)

    def run(  # type: ignore[reportIncompatibleMethodOverride]
        self, *, input_states: dict[str, State], **kwargs
    ) -> Generator[dict[str, ekd.SimpleFieldList]]:
        for state in super().run(input_states=input_states, **kwargs):
            output_states = {}
            for dataset, s in state.items():
                s = s.copy()  # Avoid modifying the original state
                for processor in self.post_processors[dataset]:
                    s = processor.process(s)
                output_states[dataset] = self.outputs[dataset].write_step(s)
            yield output_states
