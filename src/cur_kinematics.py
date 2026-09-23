import numpy as np
import xarray as xr

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

R_earth = 6.371e6        # Earth radius [m]
Omega = 7.2921159e-5     # Earth rotation rate [s^-1]

def coriolis_parameter(lat):
    """
    Coriolis parameter.

    Parameters
    ----------
    lat : xarray.DataArray
        Latitude in degrees.

    Returns
    -------
    f : xarray.DataArray
        Coriolis parameter [s^-1].
    """

    lat_rad = np.deg2rad(lat)

    f = 2 * Omega * np.sin(lat_rad)

    return f

def relative_vorticity(u, v, lat_name="lat", lon_name="lon"):
    """
    Compute vertical relative vorticity on a lat/lon grid.

        zeta = curl_h(u, v)

    Uses spherical-coordinate derivatives.

    Returns
    -------
    zeta : xarray.DataArray
        Relative vorticity [s^-1].
    """

    lat = u[lat_name]
    lat_rad = np.deg2rad(lat)

    coslat = np.cos(lat_rad)

    # differentiate() is with respect to degrees, so convert
    # derivatives from per-degree to per-radian
    dv_dlon = v.differentiate(lon_name) * (180.0 / np.pi)

    ucos = u * coslat
    ducos_dlat = ucos.differentiate(lat_name) * (180.0 / np.pi)

    zeta = (
        dv_dlon - ducos_dlat
    ) / (R_earth * coslat)

    zeta.name = "vorticity"
    zeta.attrs["long_name"] = "vertical relative vorticity"
    zeta.attrs["units"] = "s^-1"

    return zeta

def horizontal_divergence(u, v, lat_name="lat", lon_name="lon"):
    """
    Compute horizontal divergence on a lat/lon grid.

        div = div_h(u, v)

    Uses spherical-coordinate derivatives.

    Returns
    -------
    div : xarray.DataArray
        Horizontal divergence [s^-1].
    """

    lat = u[lat_name]
    lat_rad = np.deg2rad(lat)

    coslat = np.cos(lat_rad)

    # differentiate() is with respect to degrees, so convert
    # derivatives from per-degree to per-radian
    du_dlon = u.differentiate(lon_name) * (180.0 / np.pi)

    vcos = v * coslat
    dvcos_dlat = vcos.differentiate(lat_name) * (180.0 / np.pi)

    div = (
        du_dlon + dvcos_dlat
    ) / (R_earth * coslat)

    div.name = "divergence"
    div.attrs["long_name"] = "horizontal divergence"
    div.attrs["units"] = "s^-1"

    return div

def compute_for_ds(
    ds,
    u_name="u",
    v_name="v",
    lat_name="lat",
    lon_name="lon",
):
    """
    Compute current vorticity, divergence, and their
    Coriolis-normalized forms.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing u and v currents.

    u_name, v_name : str
        Names of the eastward and northward current components.

    lat_name, lon_name : str
        Names of latitude and longitude coordinates.

    Returns
    -------
    out : xarray.Dataset
        Dataset containing

        vorticity
        divergence
        normalized_vorticity
        normalized_divergence
        f
    """

    u = ds[u_name]
    v = ds[v_name]

    zeta = relative_vorticity(
        u,
        v,
        lat_name=lat_name,
        lon_name=lon_name,
    )

    div = horizontal_divergence(
        u,
        v,
        lat_name=lat_name,
        lon_name=lon_name,
    )

    f = coriolis_parameter(ds[lat_name])

    normalized_vorticity = zeta / f
    normalized_divergence = div / f

    normalized_vorticity.name = "normalized_vorticity"
    normalized_vorticity.attrs["long_name"] = "relative vorticity normalized by f"
    normalized_vorticity.attrs["units"] = "1"

    normalized_divergence.name = "normalized_divergence"
    normalized_divergence.attrs["long_name"] = "horizontal divergence normalized by f"
    normalized_divergence.attrs["units"] = "1"

    out = xr.Dataset(
        {
            "vorticity": zeta,
            "divergence": div,
            "normalized_vorticity": normalized_vorticity,
            "normalized_divergence": normalized_divergence,
            "f": f,
        }
    )

    return out