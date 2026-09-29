import xarray as xr
import xrft
import numpy as np

from spectral_analysis import isotropize

def scalar_coherence_spectrum(
    x,
    y,
    dims=("x", "y"),
    time_dim="time",
    detrend="linear",
    window="hann",
    decorrelation_hours=24 * 7,
    alpha=0.05,
    sample_interval_hours=None,
    nbins=None,
):
    """
    Compute spatial coherence between two scalar fields.

    For scalar fields X and Y, the isotropic magnitude-squared
    coherence is

        gamma^2(k) = |S_xy(k)|^2 / [S_xx(k) S_yy(k)]

    where

        S_xy = <conj(X_hat) Y_hat>.

    The associated phase is

        phi(k) = angle(S_xy).

    Auto- and cross-spectra are averaged over temporal realizations
    before coherence is calculated. For the isotropic result, the
    spectra are additionally averaged over annuli before coherence
    is calculated.

    Parameters
    ----------
    x, y : xr.DataArray
        Scalar fields to compare.

    dims : tuple
        Spatial dimensions for the 2-D FFT.

    time_dim : str
        Dimension over which repeated realizations are averaged.

    detrend, window :
        Passed to xrft.fft.

    decorrelation_hours : float
        Assumed decorrelation timescale used to estimate the effective
        number of independent temporal realizations.

    alpha : float
        Significance level for the approximate coherence threshold.

    sample_interval_hours : float, optional
        Sampling interval. Inferred from the time coordinate if omitted.

    nbins : int, optional
        Number of radial bins passed to annular_mean().

    Returns
    -------
    ds : xr.Dataset
        Dataset containing 2-D and isotropic auto-spectra,
        cross-spectrum, coherence, phase, transfer function,
        and approximate coherence significance threshold.
    """

    # ------------------------------------------------------------
    # Align fields
    # ------------------------------------------------------------

    x, y = xr.align(
        x,
        y,
        join="inner",
    )

    # Spatial FFT dimensions must each be one Dask chunk
    chunks = {
        dims[0]: -1,
        dims[1]: -1,
    }

    x = x.chunk(chunks)
    y = y.chunk(chunks)

    fft_kwargs = dict(
        dim=list(dims),
        detrend=detrend,
        window=window,
        prefix="k",
        true_phase=True,
        true_amplitude=False,
    )

    # ------------------------------------------------------------
    # Fourier transforms
    # ------------------------------------------------------------

    print("Computing FFTs")

    X = xrft.fft(
        x,
        **fft_kwargs,
    )

    Y = xrft.fft(
        y,
        **fft_kwargs,
    )

    # ------------------------------------------------------------
    # Auto- and cross-spectra for each time realization
    #
    # Convention:
    #     S_xy = conj(X) * Y
    # ------------------------------------------------------------

    S_xx_t = (X.conj() * X).real
    S_yy_t = (Y.conj() * Y).real
    S_xy_t = X.conj() * Y

    # ------------------------------------------------------------
    # Average over time BEFORE forming coherence
    # ------------------------------------------------------------

    print("Averaging temporal realizations")

    spectra = xr.Dataset(
        {
            "S_xx": S_xx_t.mean(time_dim),
            "S_yy": S_yy_t.mean(time_dim),
            "S_xy": S_xy_t.mean(time_dim),
        }
    ).compute()

    # ------------------------------------------------------------
    # 2-D coherence and phase
    # ------------------------------------------------------------

    print("Computing 2-D coherence")

    denominator_2d = (
        spectra.S_xx
        * spectra.S_yy
    )

    coherence_2d = (
        np.abs(spectra.S_xy) ** 2
        / denominator_2d
    ).where(
        denominator_2d > 0
    ).clip(0, 1)

    phase_2d = xr.apply_ufunc(
        np.angle,
        spectra.S_xy,
    )

    # ------------------------------------------------------------
    # Isotropic spectra
    #
    # Average spectra over annuli FIRST, then calculate coherence.
    # ------------------------------------------------------------

    print("Isotropizing")

    S_xx_iso = isotropize(
        spectra.S_xx,
        nbins=nbins,
    )

    S_yy_iso = isotropize(
        spectra.S_yy,
        nbins=nbins,
    )

    S_xy_iso = isotropize(
        spectra.S_xy,
        nbins=nbins,
    )

    denominator_iso = (
        S_xx_iso
        * S_yy_iso
    )

    coherence_iso = (
        np.abs(S_xy_iso) ** 2
        / denominator_iso
    ).where(
        denominator_iso > 0
    ).clip(0, 1)

    phase_iso = xr.apply_ufunc(
        np.angle,
        S_xy_iso,
    )

    # ------------------------------------------------------------
    # Scalar transfer function
    #
    # Y_hat ~= H X_hat
    # ------------------------------------------------------------

    transfer_iso = (
        S_xy_iso / S_xx_iso
    ).where(
        S_xx_iso > 0
    )

    transfer_gain_iso = np.abs(
        transfer_iso
    )

    # ------------------------------------------------------------
    # Approximate significance threshold based on temporal
    # realizations only
    # ------------------------------------------------------------

    if sample_interval_hours is None:

        time_values = x[time_dim].values

        sample_interval_hours = float(
            np.median(np.diff(time_values))
            / np.timedelta64(1, "h")
        )

    total_hours = (
        x.sizes[time_dim]
        * sample_interval_hours
    )

    n_effective = (
        total_hours
        / decorrelation_hours
    )

    if n_effective <= 1:
        raise ValueError(
            "Fewer than two effective independent realizations."
        )

    coherence_CL = (
        1
        - alpha ** (1 / (n_effective - 1))
    )

    phase_iso_significant = phase_iso.where(
        coherence_iso >= coherence_CL
    )

    # ------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------

    ds = xr.Dataset(
        {
            "S_xx_2d": spectra.S_xx,
            "S_yy_2d": spectra.S_yy,
            "S_xy_2d": spectra.S_xy,

            "coherence_2d": coherence_2d,
            "phase_2d": phase_2d,

            "S_xx_iso": S_xx_iso,
            "S_yy_iso": S_yy_iso,
            "S_xy_iso": S_xy_iso,

            "coherence_iso": coherence_iso,
            "phase_iso": phase_iso,
            "phase_iso_significant": phase_iso_significant,

            "transfer_iso": transfer_iso,
            "transfer_gain_iso": transfer_gain_iso,

            "coherence_CL": xr.DataArray(
                coherence_CL
            ),
        }
    )

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    ds.coherence_2d.attrs["units"] = "1"
    ds.coherence_iso.attrs["units"] = "1"

    ds.phase_2d.attrs["units"] = "radians"
    ds.phase_iso.attrs["units"] = "radians"
    ds.phase_iso_significant.attrs["units"] = "radians"

    ds.coherence_CL.attrs = {
        "long_name": (
            "approximate temporal-only coherence significance threshold"
        ),
        "alpha": alpha,
        "confidence_level": 1 - alpha,
        "effective_independent_averages": n_effective,
        "effective_degrees_of_freedom": 2 * n_effective,
        "decorrelation_hours": decorrelation_hours,
        "sample_interval_hours": sample_interval_hours,
    }

    ds.attrs = {
        "field_type": "scalar",
        "cross_spectrum_convention": (
            "S_xy = conjugate(X_hat) * Y_hat"
        ),
        "phase_definition": "angle(S_xy)",
        "coherence_definition": (
            "|S_xy|^2 / (S_xx * S_yy)"
        ),
        "isotropization": (
            "auto- and cross-spectra annularly averaged "
            "before coherence is calculated"
        ),
        "window": window,
        "detrend": detrend,
        "decorrelation_hours": decorrelation_hours,
    }

    return ds