# (C) Copyright 2024-ECMWF
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from io import BytesIO

from anemoi.inference.types import FloatArray
from anemoi.inference.types import State

from anemoi.inference.grib.encoding import grib_keys
from anemoi.inference.grib.templates.manager import TemplateManager
from anemoi.inference.grib.encoding import encode_message
from anemoi.inference.output import Output

LOG = logging.getLogger(__name__)


class GribOutput(Output):
    """Handles grib."""

    def __init__(
        self,
        context,
        target: BytesIO,
        *,
        encoding: Optional[Dict[str, Any]] = None,
        templates: Optional[Union[List[str], str]] = None,
        grib1_keys: Optional[Dict[str, Any]] = None,
        grib2_keys: Optional[Dict[str, Any]] = None,
        variables: Optional[List[str]] = None,
    ) -> None:
        """Initialize the GribOutput object.

        Parameters
        ----------
        context
            Runner context.
        target : BytesIO
            The target output stream for the grib messages.
        encoding : dict, optional
            The encoding dictionary, by default None.
        templates : list or str, optional
            The templates list or string, by default None.
        grib1_keys : dict, optional
            The grib1 keys dictionary, by default None.
        grib2_keys : dict, optional
            The grib2 keys dictionary, by default None.
        output_frequency : int, optional
            The frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default None.
        variables : list, optional
            The list of variables, by default None.
        """

        super().__init__(context)
        self._first = True
        self.target = target
        self.typed_variables = self.checkpoint.typed_variables
        self.encoding = encoding if encoding is not None else {}
        self.grib1_keys = grib1_keys if grib1_keys is not None else {}
        self.grib2_keys = grib2_keys if grib2_keys is not None else {}

        self.variables = variables

        self.ensemble = False
        for d in (self.grib1_keys, self.grib2_keys, self.encoding):
            if "eps" in d:
                self.ensemble = d["eps"]
                break
            if d.get("type") in ("pf", "cf"):
                self.ensemble = True
                break

        self.template_manager = TemplateManager(self, templates)

    def write_step(self, state: State) -> None:
        """Write a step of the state.

        Parameters
        ----------
        state : State
            The state object.
        """

        reference_date = self.reference_date or self.context.reference_date
        step = state["step"]
        previous_step = state.get("previous_step")
        start_steps = state.get("start_steps", {})

        out_vars = self.variables if self.variables is not None else state["fields"].keys()
        for name in out_vars:
            values = state["fields"][name]
            keys = {}

            variable = self.typed_variables[name]

            if variable.is_computed_forcing:
                continue

            param = variable.grib_keys.get("param", name)

            template = self.template(state, name)

            keys.update(self.encoding)

            keys = grib_keys(
                values=values,
                template=template,
                date=int(reference_date.strftime("%Y%m%d")),
                time=reference_date.hour * 100,
                step=step,
                param=param,
                variable=variable,
                ensemble=self.ensemble,
                keys=keys,
                grib1_keys=self.grib1_keys,
                grib2_keys=self.grib2_keys,
                previous_step=previous_step,
                start_steps=start_steps,
            )

            for modifier in self.modifiers:
                values, template, keys = modifier(values, template, keys)

            if LOG.isEnabledFor(logging.DEBUG):
                LOG.info("Encoding GRIB %s\n%s", template, json.dumps(keys, indent=4))

            try:
                self.write_message(values, template=template, **keys)
            except Exception:
                LOG.error("Error writing field %s", name)
                LOG.error("Template: %s", template)
                LOG.error("Keys:\n%s", json.dumps(keys, indent=4, default=str))
                raise

    def write_message(self, message: FloatArray, template, **kwargs: Any) -> None:
        """Write a message to the grib file.

        Parameters
        ----------
        message : FloatArray
            The message array.
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        handle = encode_message(
            values=message,
            check_nans=True,
            metadata=kwargs,
            template=template,
        )
        handle.write(self.target)

    def template(self, state: State, name: str) -> object:
        """Get the template for a variable.

        Parameters
        ----------
        state : State
            The state object.
        name : str
            The variable name.

        Returns
        -------
        object
            The template object.
        """

        if self.template_manager is None:
            self.template_manager = TemplateManager(self, self.templates)

        return self.template_manager.template(name, state)

    def template_lookup(self, name: str) -> dict:
        """Lookup the template for a variable.

        Parameters
        ----------
        name : str
            The variable name.

        Returns
        -------
        dict
            The template dictionary.
        """
        return self.encoding


