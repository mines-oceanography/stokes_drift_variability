
import xarray as xr
import xrft

def power_spectrum(da):
    # FFT dimensions must each be contained in one Dask chunk
    da = da.chunk({
        "time": min(time_block, da.sizes["time"]),
        "x": -1,
        "y": -1,
    })

    return xrft.power_spectrum(
        da,
        dim=["x", "y"],
        window="hann",
        detrend="linear",
        scaling="density",
        window_correction=True,
    ).rename({
        "freq_x": "kx",
        "freq_y": "ky",
    })

def compute_spectra(ds):
    output = xr.Dataset()

    # Scalar fields
    for var in SCALAR_FIELDS:
        if var not in ds:
            continue

        spec = power_spectrum(ds[var])
        spec.name = f"{var}_PSD"
        spec.attrs["long_name"] = f"2-D power spectrum of {var}"
        spec.attrs["units"] = f"({ds[var].attrs.get('units', 'unknown')})^2 km^2"

        output[spec.name] = spec

    # Vector fields
    for field, (u_name, v_name) in VECTOR_FIELDS.items():
        if u_name not in ds or v_name not in ds:
            continue

        u_spec = power_spectrum(ds[u_name])
        v_spec = power_spectrum(ds[v_name])
        ke_spec = 0.5 * (u_spec + v_spec)

        units = ds[u_name].attrs.get("units", "unknown")
        spec_units = f"({units})^2 km^2"

        output[f"{field}_U_PSD"] = u_spec
        output[f"{field}_V_PSD"] = v_spec
        output[f"{field}_KE_PSD"] = ke_spec

        output[f"{field}_U_PSD"].attrs = {
            "long_name": f"{field} U-component power spectrum",
            "units": spec_units,
        }

        output[f"{field}_V_PSD"].attrs = {
            "long_name": f"{field} V-component power spectrum",
            "units": spec_units,
        }

        output[f"{field}_KE_PSD"].attrs = {
            "long_name": f"{field} kinetic-energy spectrum",
            "units": spec_units,
            "definition": "0.5 * (U_PSD + V_PSD)",
        }

    output["kx"].attrs["units"] = "cycles km-1"
    output["ky"].attrs["units"] = "cycles km-1"

    return output