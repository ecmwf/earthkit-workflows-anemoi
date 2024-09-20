import array_api_compat
from meters import ResourceMeter
from typing import TypeAlias

from earthkit.meteo.extreme import array as extreme
from earthkit.meteo.stats import array as stats

from cascade.backends import num_args

from earthkit.data.sources.array_list import ArrayFieldList
from earthkit.data.readers.grib.metadata import StandAloneGribMetadata
from earthkit.data.readers.grib.output import GribCoder


# from anemoicascade.wrappers.metadata import StandAloneGribMetadata
# from anemoicascade.wrappers.array_list import ArrayFieldList

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


def make_field_list(array, template, **kwargs):
    """Make a ArrayFieldList from array, template and kwargs using GribCoder.encode"""
    handle = GribCoder(template = template).encode(None, **kwargs)
    meta = StandAloneGribMetadata(handle)
    return ArrayFieldList(array, meta)


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

    def take(
        array: ArrayFieldList,
        indices: int | tuple,
        *,
        dim: int | str,
        method: str = "slice",
        **kwargs,
    ):
        if method == "slice":
            if dim != 0:
                raise ValueError("Can not slice from FieldList along dim != 0")
            if isinstance(indices, int):
                indices = [indices]
            ret = array[indices]
        else:
            if not isinstance(dim, str):
                raise ValueError(
                    "To perform isel/sel on FieldList, dim must be a string"
                )
            if method == "isel":
                ret = array.isel(**{dim: indices}, **kwargs)
            elif method == "sel":
                ret = array.sel(**{dim: indices}, **kwargs)
            else:
                raise ValueError(f"Invalid method {method}")
        
        return ArrayFieldList(ret.values, ret.metadata())

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

    def threshold(
        arr: ArrayFieldList,
        comparison: str,
        value: float,
        *,
        metadata: Metadata = None,
    ) -> ArrayFieldList:
        with ResourceMeter("THRESHOLD"):
            xp = array_api_compat.array_namespace(arr.values)
            # Find all locations where nan appears as an ensemble value
            is_nan = xp.isnan(arr.values)
            thesh = comp_str2func(xp, comparison)(arr.values, value)
            res = xp.where(is_nan, xp.nan, thesh)
            return new_fieldlist(res, arr.metadata(), resolve_metadata(metadata, arr))

    # def efi(
    #     clim: ArrayFieldList,
    #     ens: ArrayFieldList,
    #     eps: float,
    #     *,
    #     metadata: Metadata = None,
    # ) -> ArrayFieldList:
    #     with ResourceMeter(f"EFI, clim {clim.values.shape}, ens {ens.values.shape}"):
    #         xp = array_api_compat.array_namespace(ens.values, clim.values)
    #         with PatchModule(extreme, "numpy", xp):
    #             res = extreme.efi(clim.values, ens.values, eps)
    #         return new_fieldlist(
    #             res,
    #             [ens[0].metadata()],
    #             {**resolve_metadata(metadata, clim, ens), **grib.efi(clim, ens)},
    #         )

    # def sot(
    #     clim: ArrayFieldList,
    #     ens: ArrayFieldList,
    #     number: int,
    #     eps: float,
    #     *,
    #     metadata: Metadata = None,
    # ) -> ArrayFieldList:
    #     with ResourceMeter("SOT"):
    #         xp = array_api_compat.array_namespace(ens.values, clim.values)
    #         with PatchModule(extreme, "numpy", xp):
    #             res = extreme.sot(clim.values, ens.values, number, eps)
    #         return new_fieldlist(
    #             res,
    #             [ens[0].metadata()],
    #             {
    #                 **resolve_metadata(metadata, clim, ens),
    #                 **grib.sot(clim, ens, number),
    #             },
    #         )

    # def quantiles(
    #     ens: ArrayFieldList, quantile: float, *, metadata: Metadata = None
    # ) -> ArrayFieldList:
    #     with ResourceMeter("QUANTILES"):
    #         xp = array_api_compat.array_namespace(ens.values)
    #         with PatchModule(stats, "numpy", xp):
    #             res = list(
    #                 stats.iter_quantiles(ens.values, [quantile], method="numpy")
    #             )[0]
    #         return new_fieldlist(
    #             res,
    #             [ens[0].metadata()],
    #             {**resolve_metadata(metadata, ens), "perturbationNumber": quantile},
    #         )

    def filter(
        arr1: ArrayFieldList,
        arr2: ArrayFieldList,
        comparison: str,
        threshold: float,
        *,
        replacement: float = 0,
        metadata: Metadata = None,
    ) -> ArrayFieldList:
        with ResourceMeter("FILTER"):
            xp = array_api_compat.array_namespace(arr1.values, arr2.values)
            condition = comp_str2func(xp, comparison)(arr2.values, threshold)
            res = xp.where(condition, replacement, arr1.values)
            return new_fieldlist(
                res, arr1.metadata(), resolve_metadata(metadata, arr1, arr2)
            )

    # def pca(
    #     config,
    #     ens: ArrayFieldList,
    #     spread: ArrayFieldList,
    #     mask: np.ndarray,
    #     target: str,
    # ):
    #     xp = array_api_compat.array_namespace(ens.values)
    #     lat_lon = ens.to_latlon(flatten=True)
    #     ens_data = xp.reshape(
    #         ens.to_array(flatten=True),
    #         (config.num_members, len(config.steps), len(lat_lon["lat"])),
    #     )
    #     with ResourceMeter(
    #         f"PCA: lat {lat_lon['lat'].shape} lon {lat_lon['lon'].shape}"
    #     ):
    #         pca_data = clustereps.pca.do_pca(
    #             config,
    #             lat_lon["lat"],
    #             normalise_angles(lat_lon["lon"]),
    #             ens_data,
    #             spread[0].values,
    #             mask,
    #         )

    #     ## Save data
    #     if target is not None:
    #         np.savez_compressed(target, **pca_data)

    #     return (pca_data, ens[0].metadata())

    # def cluster(
    #     config,
    #     pca_output: tuple[dict, GribMetadata],
    #     ncomp_file: str,
    #     indexes: str,
    #     deterministic: str,
    # ):
    #     pca_data, pca_metadata = pca_output

    #     ## Compute number of PCs based on the variance threshold
    #     var_cum = pca_data["var_cum"]
    #     npc = config.npc
    #     if npc <= 0:
    #         npc = clustereps.cluster.select_npc(config.var_th, var_cum)
    #         if ncomp_file is not None:
    #             with open(ncomp_file, "w") as f:
    #                 print(npc, file=f)

    #     print(f"Number of PCs used: {npc}, explained variance: {var_cum[npc-1]} %")

    #     with ResourceMeter("Clustering"):
    #         (
    #             ind_cl,
    #             centroids,
    #             rep_members,
    #             centroids_gp,
    #             rep_members_gp,
    #             ens_mean,
    #         ) = clustereps.cluster.do_clustering(
    #             config, pca_data, npc, verbose=True, dump_indexes=indexes
    #         )

    #     ## Find the deterministic forecast
    #     if deterministic is not None:
    #         with ResourceMeter("Find deterministic"):
    #             det = read_steps_grib(config.sources, deterministic, config.steps)
    #             det_index = clustereps.cluster.find_cluster(
    #                 det,
    #                 ens_mean,
    #                 pca_data["eof"][:npc, ...],
    #                 pca_data["weights"],
    #                 centroids,
    #             )
    #     else:
    #         det_index = 0

    #     metadata = pca_metadata.override(
    #         {"rep_members": rep_members, "det_index": det_index, "ind_cl": ind_cl}
    #     )
    #     ret = {
    #         "centroids": [
    #             np.array(centroids_gp),
    #             metadata.override({"type": "cm", "scenario": "centroids"}),
    #         ],
    #         "representatives": [
    #             np.array(rep_members_gp),
    #             metadata.override({"type": "cr", "scenario": "representatives"}),
    #         ],
    #     }
    #     return ret

    # def attribution(config, scenario: str, pca_output: dict, cluster_output: dict):
    #     pca_data, _ = pca_output
    #     scdata, metadata = cluster_output[scenario]

    #     with ResourceMeter("Read climatology"):
    #         ## Read climatology fields
    #         clim = clustereps.attribution.get_climatology_fields(
    #             config.climMeans, config.seasons, config.stepDate
    #         )

    #         ## Read climatological EOFs
    #         clim_eof, clim_ind = clustereps.attribution.get_climatology_eof(
    #             config.climClusterCentroidsEOF,
    #             config.climEOFs,
    #             config.climPCs,
    #             config.climSdv,
    #             config.climClusterIndex,
    #             config.nClusterClim,
    #             config.monStartDoS,
    #             config.monEndDoS,
    #         )

    #     weights = pca_data["weights"]

    #     ## Compute anomalies
    #     anom = scdata - clim
    #     anom = np.clip(anom, -config.clip, config.clip)

    #     with ResourceMeter(f"Attribute {scenario}"):
    #         cluster_att, min_dist = clustereps.attribution.attribution(
    #             anom, clim_eof, clim_ind, weights
    #         )

    #     return (metadata, scdata, anom, cluster_att, min_dist)

    # def retrieve(request: dict | list[dict], **kwargs):
    #     with ResourceMeter(f"RETRIEVE {request}, {kwargs}"):
    #         res = ek_retrieve(request, **kwargs)
    #         ret = ArrayFieldList(
    #             res.values,
    #             [GribMetadata(metadata._handle) for metadata in res.metadata()],
    #         )
    #         return ret

    def set_metadata(data: ArrayFieldList, metadata: dict) -> ArrayFieldList:

        return new_fieldlist(data.values, data.metadata(), metadata)

    # def write(data: ArrayFieldList, loc, metadata: dict | None = None):
    #     if loc == "null:":
    #         return
    #     target = target_from_location(loc)
    #     if isinstance(target, (FileTarget, FileSetTarget)):
    #         # Allows file to be appended on each write call
    #         target.enable_recovery()
    #     assert len(data) == 1, f"Expected single field, received {len(data)}"

    #     template = data.metadata()[0]
    #     if metadata is not None:
    #         template = template.override(metadata)

    #     with ResourceMeter(f"WRITE {loc}"):
    #         write_grib(target, template._handle, data[0].values)

    # def cluster_write(
    #     config,
    #     scenario,
    #     attribution_output,
    #     cluster_dests,
    # ):
    #     metadata, scdata, anom, cluster_att, min_dist = attribution_output
    #     cluster_type, ind_cl, rep_members, det_index = [
    #         metadata.get(x) for x in ["type", "ind_cl", "rep_members", "det_index"]
    #     ]

    #     keys, steps = get_output_keys(config, metadata)
    #     with ResourceMeter(f"WRITE {scenario}"):
    #         ## Write anomalies and cluster scenarios
    #         dest, adest = cluster_dests
    #         target = target_from_location(dest)
    #         anom_target = target_from_location(adest)
    #         keys["type"] = cluster_type
    #         write_cluster_attr_grib(
    #             steps,
    #             ind_cl,
    #             rep_members,
    #             det_index,
    #             scdata,
    #             anom,
    #             cluster_att,
    #             target,
    #             anom_target,
    #             keys,
    #             ncl_dummy=config.ncl_dummy,
    #         )

    #         ## Write report output
    #         # table: attribution cluster index for all fc clusters, step
    #         np.savetxt(
    #             pjoin(
    #                 config.output_root,
    #                 f"{config.step_start}_{config.step_end}dist_index_{scenario}.txt",
    #             ),
    #             min_dist,
    #             fmt="%-10.5f",
    #             delimiter=3 * " ",
    #         )

    #         # table: distance measure for all fc clusters, step
    #         np.savetxt(
    #             pjoin(
    #                 config.output_root,
    #                 f"{config.step_start}_{config.step_end}att_index_{scenario}.txt",
    #             ),
    #             cluster_att,
    #             fmt="%-3d",
    #             delimiter=3 * " ",
    #         )
