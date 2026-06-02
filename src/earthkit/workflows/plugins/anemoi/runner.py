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
from datetime import datetime
from io import BytesIO
from typing import Literal

import earthkit.data as ekd
from anemoi.inference.config.utils import input_types_config, multi_datasets_config
from anemoi.inference.input import Input
from anemoi.inference.inputs import create_input
from anemoi.inference.metadata import Metadata
from anemoi.inference.output import Output
from anemoi.inference.outputs.gribmemory import GribMemoryOutput
from anemoi.inference.runners import Runner
from anemoi.inference.variables import Variables
from anemoi.utils.dates import frequency_to_seconds
from anemoi.utils.grib import shortname_to_paramid

from .types import ENSEMBLE_DIMENSION_NAME

LOG = logging.getLogger(__name__)


class CascadeOutput(Output):
    """Custom output class for the CascadeRunner that converts model states to GRIB format and then to earthkit FieldList."""

    def __init__(self, runner: CascadeRunner, metadata: Metadata):
        super().__init__(runner, metadata)

        self._grib_memory = BytesIO()
        self.grib_output = GribMemoryOutput(runner, metadata, out=self._grib_memory)

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

            self.grib_output.write_state(state)
            self._grib_memory.seek(0, 0)
            fieldlist: ekd.SimpleFieldList = ekd.from_source("stream", self._grib_memory, read_all=True)  # type: ignore[reportAssignmentType]
            self._grib_memory.seek(0, 0)

            return fieldlist

        except Exception:
            LOG.error("Error converting state to grib, will convert to ArrayField.", exc_info=True)

        import numpy as np

        fields = []

        step = frequency_to_seconds(state["date"] - initial_date) // 3600

        for var, array in state["fields"].items():
            variable = self.typed_variables[var]
            paramId = shortname_to_paramid(variable.param)

            grib_metadata.update(
                {
                    "step": step,
                    "base_datetime": initial_date,
                    "valid_datetime": state["date"],
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

    def create_input(
        self,
        input_type: Literal["prognostics", "constant_forcings", "dynamic_forcings", "boundary_forcings"],
        dataset_name: str,
        metadata: Metadata,
    ) -> Input:
        variables = Variables(metadata)
        match input_type:
            case "prognostics":
                variables = (
                    variables.default_input_variables()
                )  # Only difference to parent class, to get all variables as prognostics
                config = input_types_config(self.config, "prognostic_input", "input") if variables else "empty"  # type: ignore[reportArgumentType]
            case _:
                return super().create_input(input_type, dataset_name, metadata)

        config = multi_datasets_config(config, dataset_name, self.dataset_names)
        input = create_input(self, config, metadata, variables=variables, purpose=input_type)  # type: ignore[reportArgumentType]

        LOG.info(f"[{dataset_name}] {input_type.replace('_', ' ').capitalize()} input: {input}")
        return input

    def create_output(self, dataset_name: str, metadata: Metadata) -> CascadeOutput:
        output = CascadeOutput(self, metadata)  # Force the use of CascadeOutput for all datasets
        LOG.info(f"[{dataset_name}] Output: {output}")
        return output
