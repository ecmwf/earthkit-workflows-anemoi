"""
Provide a connection between anemoi and cascade.

Exposes the `fluent` API for running anemoi models in cascade.
"""

from anemoicascade import fluent
from anemoicascade import serialisation

from cascade.backends import register as register_backend

# from .wrappers.array_list import ArrayFieldList
from .backends.fieldlist import ArrayFieldListBackend

from earthkit.data.sources import array_list
register_backend(array_list.ArrayFieldList, ArrayFieldListBackend)
