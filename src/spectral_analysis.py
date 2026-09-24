import xarray as xr
import xrft


def power_spectrum(da):
    """Compute the 2-D spatial power spectrum of a DataArray."""

    # Spatial FFT dimensions must each be contained in one Dask chunk.
    # Time has already been blocked by the calling analysis script.
    da = da.chunk({
        "time": -1,
        "x": -1,
        "y": -1,
    })

    spectrum = xrft.power_spectrum(
        da,
        dim=["x", "y"],
        window="hann",
        detrend="linear",
        scaling="density",
        window_correction=True,
    )

    return spectrum.rename({
        "freq_x": "kx",
        "freq_y": "ky",
    })


def compute_spectra(ds):
    """
    Compute the 2-D spatial power spectrum of each variable in a Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing variables on an x-y grid.

    Returns
    -------
    xarray.Dataset
        Dataset containing the 2-D power spectrum of each input variable.
    """

    output = xr.Dataset()

    for var in ds.data_vars:

        spec = power_spectrum(ds[var])

        spec.name = f"{var}_PSD"
        spec.attrs = {
            "long_name": f"2-D power spectrum of {var}",
            "units": f"({ds[var].attrs.get('units', 'unknown')})^2 km^2",
        }

        output[spec.name] = spec

    output["kx"].attrs["units"] = "cycles km-1"
    output["ky"].attrs["units"] = "cycles km-1"

    return output