import xarray as xr
import xrft
import numpy as np
from scipy.stats.distributions import chi2

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

def isotropize(
    da,
    kx_name="kx",
    ky_name="ky",
    nbins=None,
):
    """
    Annular mean of a real or complex 2-D Fourier-space field.

    Only the fully sampled circular region of Fourier space is used.
    The zero-wavenumber mode is excluded.
    """

    da = da.transpose(
        ky_name,
        kx_name,
    ).compute()

    kx = da[kx_name].values
    ky = da[ky_name].values

    KX, KY = np.meshgrid(
        kx,
        ky,
    )

    KR = np.sqrt(
        KX**2 + KY**2
    )

    # ------------------------------------------------------------
    # Fundamental Fourier spacings
    # ------------------------------------------------------------

    dkx = np.min(
        np.diff(np.sort(np.unique(kx)))
    )

    dky = np.min(
        np.diff(np.sort(np.unique(ky)))
    )

    # Characteristic radial-bin width
    dk = max(
        abs(dkx),
        abs(dky),
    )

    # ------------------------------------------------------------
    # Maximum radius with complete azimuthal sampling
    # ------------------------------------------------------------

    kr_max = min(
        np.abs(kx).max(),
        np.abs(ky).max(),
    )

    # ------------------------------------------------------------
    # Radial bins
    #
    # First bin is centered at dk rather than dk/2.
    # This also excludes the zero mode.
    # ------------------------------------------------------------

    if nbins is None:

        centers = np.arange(
            dk,
            kr_max + 0.5 * dk,
            dk,
        )

        bins = np.concatenate(
            [
                [0.5 * dk],
                centers + 0.5 * dk,
            ]
        )

    else:

        bins = np.linspace(
            0.5 * dk,
            kr_max,
            nbins + 1,
        )

        centers = 0.5 * (
            bins[:-1]
            + bins[1:]
        )

    # ------------------------------------------------------------
    # Bin Fourier coefficients
    # ------------------------------------------------------------

    kr_flat = KR.ravel()
    values = da.values.ravel()

    inds = np.digitize(
        kr_flat,
        bins,
    ) - 1

    dtype = (
        complex
        if np.iscomplexobj(values)
        else float
    )

    result = np.full(
        len(centers),
        np.nan,
        dtype=dtype,
    )

    for i in range(len(centers)):

        mask = (
            (inds == i)
            & np.isfinite(values)
            & (kr_flat <= kr_max)
        )

        if mask.any():

            result[i] = values[mask].mean()

    return xr.DataArray(
        result,
        dims="kr",
        coords={
            "kr": centers,
        },
    )

def CI(da_spectra: xr.DataArray, M: float, alpha: float = 0.05):
    """
    Chi-square CI for spectra.
    
    """
    # Degrees of freedom (scalar)
    nu = 2 * M

    # Error factors (scalars)
    err_low  = nu / chi2.ppf(1 - alpha/2, df=nu)
    err_high = nu / chi2.ppf(alpha/2, df=nu)

    # Broadcast automatically across spectrum dims
    CI_low  = da_spectra * err_low
    CI_high = da_spectra * err_high

    return xr.Dataset(
        {
            "spectrum": da_spectra,
            "CI_low": CI_low,
            "CI_high": CI_high,
            "nu": nu,
            "M": M
        }
    )

def variance_lost(
    full,
    experiments,
    kdim=None,
):
    """
    Quantify variance lost relative to a full simulation as a
    function of spatial scale.

    Parameters
    ----------
    full : xr.DataArray
        Isotropic PSD from the full simulation.

    experiments : dict
        Dictionary of mechanism-off spectra, e.g.

        {
            "NOREF": spectrum,
            "NOCRT": spectrum,
            "NOREL": spectrum,
            "NOADV": spectrum,
        }

    kdim : str, optional
        Radial-wavenumber dimension. If None, inferred from full.

    Returns
    -------
    xr.Dataset

    Variables
    ---------
    spectral_loss
        P_full - P_experiment at each k.

    spectral_loss_fraction
        100 * (P_full - P_experiment) / P_full.

    cumulative_full
        Variance in the full simulation at scales <= wavelength.

    cumulative_experiment
        Same for each experiment.

    cumulative_loss
        Integrated variance lost at scales <= wavelength.

    cumulative_loss_percent
        Percentage of full variance lost at scales <= wavelength.
    """

    import numpy as np
    import xarray as xr

    if kdim is None:
        kdim = full.dims[0]

    # ----------------------------------------------------------
    # Make sure k is positive and ordered low -> high
    # ----------------------------------------------------------

    full = full.where(full[kdim] > 0, drop=True)
    full = full.sortby(kdim)

    k = full[kdim].values

    names = []

    spectral_loss_all = []
    spectral_loss_fraction_all = []

    cumulative_full_all = []
    cumulative_exp_all = []
    cumulative_loss_all = []
    cumulative_loss_percent_all = []

    # ----------------------------------------------------------
    # Helper: integral from each k_i -> k_max
    # ----------------------------------------------------------

    def integrate_from_high_k(y, x):

        # trapezoidal contribution between neighboring points
        segment = (
            0.5
            * (y[:-1] + y[1:])
            * np.diff(x)
        )

        # Reverse cumulative sum:
        # integral from k_i to k_max
        cumulative = np.zeros_like(y, dtype=float)

        cumulative[:-1] = np.cumsum(
            segment[::-1]
        )[::-1]

        return cumulative

    # Full cumulative variance only depends on full spectrum
    full_values = full.values

    cumulative_full = integrate_from_high_k(
        full_values,
        k,
    )

    # ----------------------------------------------------------
    # Each mechanism experiment
    # ----------------------------------------------------------

    for name, exp in experiments.items():

        full_aligned, exp_aligned = xr.align(
            full,
            exp,
            join="inner",
        )

        # Re-grab k after alignment
        kk = full_aligned[kdim].values

        pf = full_aligned.values
        pe = exp_aligned.values

        # ----------------------------------------------
        # Local spectral loss
        # ----------------------------------------------

        spectral_loss = pf - pe

        spectral_loss_fraction = (
            100 * spectral_loss / pf
        )

        # ----------------------------------------------
        # Cumulative variance at scales <= L
        #
        # Since L = 1/k:
        # integrate from k -> kmax
        # ----------------------------------------------

        vf = integrate_from_high_k(
            pf,
            kk,
        )

        ve = integrate_from_high_k(
            pe,
            kk,
        )

        variance_loss = vf - ve

        loss_percent = np.full_like(
            variance_loss,
            np.nan,
            dtype=float,
        )

        valid = vf > 0

        loss_percent[valid] = (
            100
            * variance_loss[valid]
            / vf[valid]
        )

        # ----------------------------------------------
        # Save
        # ----------------------------------------------

        names.append(name)

        spectral_loss_all.append(
            xr.DataArray(
                spectral_loss,
                dims=[kdim],
                coords={kdim: kk},
            )
        )

        spectral_loss_fraction_all.append(
            xr.DataArray(
                spectral_loss_fraction,
                dims=[kdim],
                coords={kdim: kk},
            )
        )

        cumulative_full_all.append(
            xr.DataArray(
                vf,
                dims=[kdim],
                coords={kdim: kk},
            )
        )

        cumulative_exp_all.append(
            xr.DataArray(
                ve,
                dims=[kdim],
                coords={kdim: kk},
            )
        )

        cumulative_loss_all.append(
            xr.DataArray(
                variance_loss,
                dims=[kdim],
                coords={kdim: kk},
            )
        )

        cumulative_loss_percent_all.append(
            xr.DataArray(
                loss_percent,
                dims=[kdim],
                coords={kdim: kk},
            )
        )

    # ----------------------------------------------------------
    # Combine experiments
    # ----------------------------------------------------------

    ds = xr.Dataset(
        {
            "spectral_loss":
                xr.concat(
                    spectral_loss_all,
                    dim="experiment",
                ),

            "spectral_loss_percent":
                xr.concat(
                    spectral_loss_fraction_all,
                    dim="experiment",
                ),

            "cumulative_full":
                xr.concat(
                    cumulative_full_all,
                    dim="experiment",
                ),

            "cumulative_experiment":
                xr.concat(
                    cumulative_exp_all,
                    dim="experiment",
                ),

            "cumulative_loss":
                xr.concat(
                    cumulative_loss_all,
                    dim="experiment",
                ),

            "cumulative_loss_percent":
                xr.concat(
                    cumulative_loss_percent_all,
                    dim="experiment",
                ),
        },
        coords={
            "experiment": names,
        },
    )

    # Wavelength corresponding to each radial wavenumber
    ds = ds.assign_coords(
        wavelength_km=1 / ds[kdim]
    )

    return ds