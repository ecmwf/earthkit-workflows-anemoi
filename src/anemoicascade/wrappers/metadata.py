import copy

from earthkit.data import from_source
from earthkit.data.readers.grib import metadata
from earthkit.data.readers.grib import memory
from earthkit.data.readers.grib import codes


class StandAloneGribMetadata(metadata.StandAloneGribMetadata):
    def __init__(self, handle, clear_data: bool = True):
        # if clear_data:
        #     # Clear data values, keeping only metadata
        #     handle = handle.clone()
        #     handle.set_array("values", handle.get_array("values").shape)
        super().__init__(handle)

    def __getstate__(self) -> dict:
        ret = self.__dict__.copy()
        ret["_handle"] = self._handle.get_buffer()
        open('byte_array_get', 'wb').write(self._handle.get_buffer())
        return ret

    def __setstate__(self, state: dict):
        open('byte_array_set', 'wb').write(state['_handle'])
        # print(from_source('memory', state['_handle']).ls())

        state["_handle"] = from_source('memory', state['_handle'])[0].handle.clone()

        self.__dict__.update(state)

    def override(self, *args, **kwargs) -> "StandAloneGribMetadata":
        raise Exception('No', args, kwargs)
        ret = super().override(*args, **kwargs)
        return StandAloneGribMetadata(ret._handle, clear_data=False)

    def _hide_internal_keys(self):
        return self