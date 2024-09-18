"""
Copied from ppcascade.wrappers.array_list.py at 9834abd9cb8748ca24dfbd392fe218a6aa51f46e
https://github.com/ecmwf/pproc-cascade/tree/9834abd9cb8748ca24dfbd392fe218a6aa51f46e
"""

from earthkit.data import FieldList
from earthkit.data.sources import array_list

from earthkit.data.readers.grib.output import GribCoder

from .metadata import StandAloneGribMetadata


class ArrayFieldList(array_list.ArrayFieldList):
    def __init__(self, array, metadata):
        if isinstance(metadata, list):
            metadata = [StandAloneGribMetadata(m._handle) for m in metadata]
        else:
            metadata = StandAloneGribMetadata(metadata._handle)
        super().__init__(array, metadata)

    def __getstate__(self) -> dict:
        return {
            "array": self.values,
            "metadata": self.metadata(),
        }

    def __setstate__(self, state: dict):
        new_fieldlist = FieldList.from_array(state["array"], state["metadata"])
        self.__dict__.update(new_fieldlist.__dict__)
        del new_fieldlist


def make_field_list(array, template, **kwargs):
    """Make a ArrayFieldList from array, template and kwargs using GribCoder.encode"""
    handle = GribCoder(template = template).encode(None, **kwargs)
    metadata = StandAloneGribMetadata(handle)
    return ArrayFieldList(array, metadata)