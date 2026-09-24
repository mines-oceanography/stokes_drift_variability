import gc
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from pyproj import Geod
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from dask.distributed import Client, LocalCluster
from tqdm.auto import tqdm

def m_to_km(x_m):
    return x_m / 1000.0

def prepare_regridding_geometry(
    ds,
    x_bounds=None,
    y_bounds=None,
):
    """
    Construct a uniform Cartesian grid and a reusable Delaunay
    triangulation for linear interpolation.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 1D latitude and longitude coordinates.
    x_bounds, y_bounds : tuple or None
        Optional target-coordinate limits in km.

    Returns
    -------
    dict
        Regridding geometry.
    """
    lat = np.asarray(ds.latitude.values)
    lon = np.asarray(ds.longitude.values)

    lon2d, lat2d = np.meshgrid(lon, lat, indexing="xy")

    geod = Geod(ellps="WGS84")

    # Distance eastward from the western edge.
    _, _, dist_x = geod.inv(
        np.full_like(lon2d, lon[0]),
        lat2d,
        lon2d,
        lat2d,
    )

    # Distance northward from the southern edge.
    _, _, dist_y = geod.inv(
        lon2d,
        np.full_like(lat2d, lat[0]),
        lon2d,
        lat2d,
    )

    x_km = m_to_km(dist_x)
    y_km = m_to_km(dist_y)

    dx = np.diff(x_km, axis=1)
    dy = np.diff(y_km, axis=0)

    # Ignore zero or negative differences, should they occur.
    dx_min = np.nanmin(dx[dx > 0])
    dy_min = np.nanmin(dy[dy > 0])

    x_uniform = np.arange(
        x_km.min(),
        x_km.max() + dx_min,
        dx_min,
    )

    y_uniform = np.arange(
        y_km.min(),
        y_km.max() + dy_min,
        dy_min,
    )

    # Crop before interpolation instead of computing pixels
    # that will later be discarded.
    if x_bounds is not None:
        x_uniform = x_uniform[
            (x_uniform >= x_bounds[0])
            & (x_uniform <= x_bounds[1])
        ]

    if y_bounds is not None:
        y_uniform = y_uniform[
            (y_uniform >= y_bounds[0])
            & (y_uniform <= y_bounds[1])
        ]

    x_new, y_new = np.meshgrid(
        x_uniform,
        y_uniform,
        indexing="xy",
    )

    source_points = np.column_stack(
        [x_km.ravel(), y_km.ravel()]
    )

    target_points = np.column_stack(
        [x_new.ravel(), y_new.ravel()]
    )

    triangulation = Delaunay(source_points)

    return {
        "triangulation": triangulation,
        "target_points": target_points,
        "x": x_uniform,
        "y": y_uniform,
        "shape": (len(y_uniform), len(x_uniform)),
        "dx_km": float(dx_min),
        "dy_km": float(dy_min),
    }

def regrid_loaded_slice(
    arrays,
    triangulation,
    target_points,
    output_shape,
):
    """
    Regrid several 2D NumPy arrays using one reusable
    source-grid triangulation.
    """
    output = {}

    for var_name, values in arrays.items():
        interpolator = LinearNDInterpolator(
            triangulation,
            np.asarray(values).ravel(),
            fill_value=np.nan,
        )

        regridded = interpolator(target_points).reshape(
            output_shape
        )

        # WW3 fields generally do not need float64 precision here.
        output[var_name] = regridded.astype(
            np.float32,
            copy=False,
        )

    return output

def regrid_dataset_to_zarr(
    ds,
    output_path,
    client,
    batch_size=8,
    x_bounds=(0, 698),
    y_bounds=(0, 698),
    overwrite=False,
    show_progress=True,
    source_file=None,
    scenario=None,
):
    """
    Regrid an xarray Dataset in small time batches and append
    each completed batch to a Zarr store.

    The full regridded dataset is never retained in RAM.
    """
    output_path = Path(output_path)

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. "
                "Use overwrite=True to replace it."
            )

        shutil.rmtree(output_path)

    variables = list(ds.data_vars)

    geometry = prepare_regridding_geometry(
        ds,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    )

    # Send the fixed geometry to workers once, rather than
    # repeatedly sending it with every task.
    tri_future = client.scatter(
        geometry["triangulation"],
        broadcast=True,
    )

    target_future = client.scatter(
        geometry["target_points"],
        broadcast=True,
    )

    n_time = ds.sizes["time"]
    first_batch = True

    iterator = range(0, n_time, batch_size)

    if show_progress:
        iterator = tqdm(
            iterator,
            total=(n_time + batch_size - 1) // batch_size,
            desc=f"Regridding {scenario or output_path.stem}",
        )

    try:
        for start in iterator:
            stop = min(start + batch_size, n_time)

            # This is the only raw-data batch loaded into the
            # notebook process at one time.
            loaded = ds.isel(
                time=slice(start, stop)
            ).load()

            futures = []

            for local_t in range(loaded.sizes["time"]):
                arrays = {
                    var: np.asarray(
                        loaded[var].isel(time=local_t).data
                    )
                    for var in variables
                }

                future = client.submit(
                    regrid_loaded_slice,
                    arrays,
                    tri_future,
                    target_future,
                    geometry["shape"],
                    pure=False,
                )

                futures.append(future)

            results = client.gather(futures)

            # Stack only the current batch.
            output_variables = {}

            for var in variables:
                values = np.stack(
                    [result[var] for result in results],
                    axis=0,
                )

                output_variables[var] = (
                    ("time", "y", "x"),
                    values,
                    dict(ds[var].attrs),
                )

            batch_ds = xr.Dataset(
                data_vars=output_variables,
                coords={
                    "time": loaded.time.values,
                    "y": geometry["y"],
                    "x": geometry["x"],
                },
            )

            batch_ds.x.attrs.update(
                {
                    "long_name": "eastward distance",
                    "units": "km",
                }
            )

            batch_ds.y.attrs.update(
                {
                    "long_name": "northward distance",
                    "units": "km",
                }
            )

            batch_ds.attrs.update(
                {
                    "source_file": (
                        str(source_file)
                        if source_file is not None
                        else ""
                    ),
                    "scenario": scenario or "",
                    "regridding_method": (
                        "piecewise linear interpolation using "
                        "a reusable Delaunay triangulation"
                    ),
                    "dx_km": geometry["dx_km"],
                    "dy_km": geometry["dy_km"],
                }
            )

            if first_batch:
                batch_ds.to_zarr(
                    output_path,
                    mode="w",
                    consolidated=False,
                )
                first_batch = False

            else:
                batch_ds.to_zarr(
                    output_path,
                    mode="a",
                    append_dim="time",
                    consolidated=False,
                )

            # Remove completed results from both the workers
            # and notebook process before starting another batch.
            # client.cancel(futures)

            del arrays
            del futures
            del results
            del output_variables
            del batch_ds
            del loaded

    finally:
        client.cancel([tri_future, target_future])
        del tri_future, target_future
        gc.collect()