import array_api_compat
from meters import ResourceMeter
from typing import TypeAlias

from earthkit.meteo.extreme import array as extreme
from earthkit.meteo.stats import array as stats

from cascade.backends import num_args

from anemoicascade.wrappers.metadata import StandAloneGribMetadata
from anemoicascade.wrappers.array_list import ArrayFieldList


def standardise_output(data):
    # Also, nest the data to avoid problems with not finding geography attribute
    if len(data.shape) == 1:
        data = data.reshape((1, *data.shape))
    assert len(data.shape) == 2
    return data


def comp_str2func(array_module, comparison: str):
    if comparison == "<=":
        return array_module.less_equal
    if comparison == "<":
        return array_module.less
    if comparison == ">=":
        return array_module.greater_equal
    return array_module.greater


Metadata: TypeAlias = "dict | callable | None"


def resolve_metadata(metadata: Metadata, *args) -> dict:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    return metadata(*args)


def new_fieldlist(data, metadata: list[StandAloneGribMetadata], overrides: dict):
    if len(overrides) > 0:
        try:
            new_metadata = [
                metadata[x].override(overrides) for x in range(len(metadata))
            ]
            return ArrayFieldList(
                standardise_output(data),
                new_metadata,
            )
        except Exception as e:
            print(
                "Error setting metadata",
                overrides,
                "edition",
                metadata[0]["edition"],
                "param",
                metadata[0]["paramId"],
            )
            print(e)
    return ArrayFieldList(standardise_output(data), metadata)


class ArrayFieldListBackend:
    def _merge(*fieldlists: list[ArrayFieldList]):
        """
        Merge fieldlist elements into a single array. fieldlists with
        different number of fields must be concatenated, otherwise, the
        elements in each fieldlist are stacked along a new dimension
        """
        if len(fieldlists) == 1:
            return fieldlists[0].values

        values = [x.values for x in fieldlists]
        xp = array_api_compat.array_namespace(*values)
        return xp.asarray(values)

    def multi_arg_function(
        func: str, *arrays: list[ArrayFieldList], metadata: Metadata = None
    ) -> ArrayFieldList:
        with ResourceMeter(func.upper()):
            merged_array = ArrayFieldListBackend._merge(*arrays)
            xp = array_api_compat.array_namespace(*merged_array)
            res = standardise_output(getattr(xp, func)(merged_array, axis=0))
            return new_fieldlist(
                res,
                [arrays[0][x].metadata() for x in range(len(res))],
                resolve_metadata(metadata, *arrays),
            )

    def two_arg_function(
        func: str, *arrays: ArrayFieldList, metadata: Metadata = None
    ) -> ArrayFieldList:
        with ResourceMeter(func.upper()):
            # First argument must be FieldList
            assert isinstance(
                arrays[0], ArrayFieldList
            ), f"Expected ArrayFieldList type, got {type(arrays[0])}"
            val1 = arrays[0].values
            if isinstance(arrays[1], ArrayFieldList):
                val2 = arrays[1].values
                metadata = resolve_metadata(metadata, *arrays)
                xp = array_api_compat.array_namespace(val1, val2)
            else:
                val2 = arrays[1]
                metadata = resolve_metadata(metadata, arrays[0])
                xp = array_api_compat.array_namespace(val1)
            res = getattr(xp, func)(val1, val2)
            return new_fieldlist(
                res, [arrays[0][x].metadata() for x in range(len(res))], metadata
            )

    def mean(
        *arrays: list[ArrayFieldList], metadata: Metadata = None
    ) -> ArrayFieldList:
        return ArrayFieldListBackend.multi_arg_function(
            "mean", *arrays, metadata=metadata
        )

    def std(*arrays: list[ArrayFieldList], metadata: Metadata = None) -> ArrayFieldList:
        return ArrayFieldListBackend.multi_arg_function(
            "std", *arrays, metadata=metadata
        )

    def min(*arrays: list[ArrayFieldList], metadata: Metadata = None) -> ArrayFieldList:
        return ArrayFieldListBackend.multi_arg_function(
            "min", *arrays, metadata=metadata
        )

    def max(*arrays: list[ArrayFieldList], metadata: Metadata = None) -> ArrayFieldList:
        return ArrayFieldListBackend.multi_arg_function(
            "max", *arrays, metadata=metadata
        )

    def sum(*arrays: list[ArrayFieldList], metadata: Metadata = None) -> ArrayFieldList:
        return ArrayFieldListBackend.multi_arg_function(
            "sum", *arrays, metadata=metadata
        )

    def prod(
        *arrays: list[ArrayFieldList], metadata: Metadata = None
    ) -> ArrayFieldList:
        return ArrayFieldListBackend.multi_arg_function(
            "prod", *arrays, metadata=metadata
        )

    def var(*arrays: list[ArrayFieldList], metadata: Metadata = None) -> ArrayFieldList:
        return ArrayFieldListBackend.multi_arg_function(
            "var", *arrays, metadata=metadata
        )

    def stack(*arrays: list[ArrayFieldList], axis: int = 0) -> ArrayFieldList:
        if axis != 0:
            raise ValueError("Can not stack FieldList along axis != 0")
        assert all(
            [len(x) == 1 for x in arrays]
        ), "Can not stack FieldLists with more than one element, use concat"
        return ArrayFieldListBackend.concat(*arrays)

    def add(*arrays: list[ArrayFieldList], metadata: Metadata = None) -> ArrayFieldList:
        return ArrayFieldListBackend.two_arg_function("add", *arrays, metadata=metadata)

    def subtract(
        *arrays: list[ArrayFieldList], metadata: Metadata = None
    ) -> ArrayFieldList:
        return ArrayFieldListBackend.two_arg_function(
            "subtract", *arrays, metadata=metadata
        )

    @num_args(2)
    def diff(
        *arrays: list[ArrayFieldList], metadata: Metadata = None
    ) -> ArrayFieldList:
        return ArrayFieldListBackend.multiply(
            ArrayFieldListBackend.subtract(*arrays, metadata=metadata), -1
        )

    def multiply(
        *arrays: list[ArrayFieldList], metadata: Metadata = None
    ) -> ArrayFieldList:
        return ArrayFieldListBackend.two_arg_function(
            "multiply", *arrays, metadata=metadata
        )

    def divide(
        *arrays: list[ArrayFieldList], metadata: Metadata = None
    ) -> ArrayFieldList:
        return ArrayFieldListBackend.two_arg_function(
            "divide", *arrays, metadata=metadata
        )

    def pow(*arrays: list[ArrayFieldList], metadata: Metadata = None) -> ArrayFieldList:
        return ArrayFieldListBackend.two_arg_function("pow", *arrays, metadata=metadata)

    def concat(*arrays: list[ArrayFieldList]) -> ArrayFieldList:
        """
        Concatenates the list of fields inside each ArrayFieldList into a single
        ArrayFieldList object

        Parameters
        ----------
        arrays: list[ArrayFieldList]
            ArrayFieldList instances to whose fields are to be concatenated

        Return
        ------
        ArrayFieldList
            Contains all fields inside the input field lists
        """
        ret = sum(arrays[1:], arrays[0])
        return ArrayFieldList(ret.values, ret.metadata())

    def take(array: ArrayFieldList, indices: int | tuple, *, axis: int):
        if axis != 0:
            raise ValueError("Can not take from FieldList along axis != 0")
        if isinstance(indices, int):
            indices = [indices]
        print(indices)
        taken = array[indices]
        return ArrayFieldList(standardise_output(taken.values), taken.metadata())

    def norm(
        *arrays: list[ArrayFieldList], metadata: Metadata = None
    ) -> ArrayFieldList:
        merged_array = ArrayFieldListBackend._merge(*arrays)
        xp = array_api_compat.array_namespace(merged_array)
        norm = standardise_output(xp.sqrt(xp.sum(xp.pow(merged_array, 2), axis=0)))
        return new_fieldlist(
            norm,
            [arrays[0][x].metadata() for x in range(len(norm))],
            resolve_metadata(metadata, *arrays),
        )

   