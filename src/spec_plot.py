import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def wavenumber_formatter(k, pos):

    if k <= 0:
        return ""

    scale_km = 1 / k

    return f"1/{scale_km:g}"

def plot_spectra_2panel(
    spectra_left,
    spectra_right,
    titles=(None, None),
    slopes=(None, None),
    figsize=(8, 5),
    xlims=(None, None),
    ylims=(None, None),
    slope_length_decades=0.5,
    slope_anchor_x=(None, None),
    slope_anchor_y=(None, None),
    ylab="Spectral Density",
    sharex=True,
    sharey=True,
    dpi=100,
    label_fontsize=21,
):
    """
    Plot two panels of isotropic spectra on log-log axes.

    Parameters
    ----------
    spectra_left, spectra_right : dict
        Plotting dictionaries for the left and right panels.

        `data` may be either:
            - xr.DataArray containing the spectrum
            - xr.Dataset containing:
                spectrum
                CI_low
                CI_high

    titles : tuple
        Titles for left and right panels.

    slopes : tuple
        Reference slopes for each panel.
        Example:
            slopes=([-3, -4], [-3])

    figsize : tuple
        Figure size.

    xlims, ylims : tuple
        Axis limits for each panel.

        Example:
            xlims=((1e-3, 1/25), (1e-3, 1/25))
            ylims=((1e-9, 1e2), (1e-9, 1e2))

    slope_length_decades : float
        Length of reference slope lines in log10 space.

    slope_anchor_x, slope_anchor_y : tuple
        Reference-slope anchor positions for each panel.

    ylab : str
        Shared y-axis label.

    sharex, sharey : bool
        Whether panels share x and/or y axes.

    dpi : int
        Figure DPI.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=dpi,
        sharex=sharex,
        sharey=sharey,
    )

    spectra_panels = [
        spectra_left,
        spectra_right,
    ]

    # ------------------------------------------------------------
    # Reciprocal wavenumber formatter
    # ------------------------------------------------------------

    def reciprocal_formatter(x, pos):
        if x <= 0:
            return ""

        denominator = 1 / x

        if np.isclose(denominator, round(denominator)):
            denominator = int(round(denominator))

        return rf"$1/{denominator:g}$"

    # ------------------------------------------------------------
    # Plot each panel
    # ------------------------------------------------------------

    for panel, ax in enumerate(axes):

        # --------------------------------------------------------
        # Panel label
        # --------------------------------------------------------

        panel_labels = ["(a)", "(b)"]

        ax.text(
            0.98,
            0.98,
            panel_labels[panel],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=label_fontsize,
            fontweight="bold",
        )

        spectra = spectra_panels[panel]

        for name, settings in spectra.items():

            data = settings["data"]

            # ----------------------------------------------------
            # DataArray or CI Dataset
            # ----------------------------------------------------

            if isinstance(data, xr.Dataset):
                da = data["spectrum"]
                CI_low = data.get("CI_low")
                CI_high = data.get("CI_high")
            else:
                da = data
                CI_low = None
                CI_high = None

            kr = da["kr"]

            valid = (
                np.isfinite(kr)
                & np.isfinite(da)
                & (kr > 0)
                & (da > 0)
            )

            kr_plot = kr.where(valid, drop=True)
            da_plot = da.where(valid, drop=True)

            color = settings.get("color")

            # ----------------------------------------------------
            # Confidence interval
            # ----------------------------------------------------

            if (
                settings.get("plot_CI", False)
                and CI_low is not None
                and CI_high is not None
            ):

                CI_valid = (
                    valid
                    & np.isfinite(CI_low)
                    & np.isfinite(CI_high)
                    & (CI_low > 0)
                    & (CI_high > 0)
                )

                kr_CI = kr.where(CI_valid, drop=True)
                CI_low_plot = CI_low.where(CI_valid, drop=True)
                CI_high_plot = CI_high.where(CI_valid, drop=True)

                ax.fill_between(
                    kr_CI.values,
                    CI_low_plot.values,
                    CI_high_plot.values,
                    color=color,
                    alpha=settings.get("CI_alpha", 0.2),
                    linewidth=0,
                )

            # ----------------------------------------------------
            # Spectrum
            # ----------------------------------------------------

            ax.loglog(
                kr_plot,
                da_plot,
                color=color,
                linestyle=settings.get("linestyle", "-"),
                linewidth=settings.get("linewidth", 1.5),
                label=settings.get("label", name),
                alpha=settings.get("alpha", 1),
            )

        # --------------------------------------------------------
        # Axes formatting
        # --------------------------------------------------------

        # ax.set_xlabel(r"$k_r$ [1/km]")

        if titles[panel] is not None:
            ax.set_title(titles[panel])

        ax.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.7,
            alpha=0.5,
            color="lightgrey",
        )

        if xlims[panel] is not None:
            ax.set_xlim(xlims[panel])

        if ylims[panel] is not None:
            ax.set_ylim(ylims[panel])

        ax.xaxis.set_major_formatter(
            FuncFormatter(reciprocal_formatter)
        )

        ax.legend(
            loc="lower left",
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
        )

    # ------------------------------------------------------------
    # Shared y-axis label
    # ------------------------------------------------------------

    axes[0].set_ylabel(ylab)

    if not sharey:
        axes[1].set_ylabel(ylab)

    # Need limits established before slope locations
    fig.canvas.draw()

    # ------------------------------------------------------------
    # Reference slopes
    # ------------------------------------------------------------

    for panel, ax in enumerate(axes):

        panel_slopes = slopes[panel]

        if not panel_slopes:
            continue

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        anchor_x = slope_anchor_x[panel]
        anchor_y = slope_anchor_y[panel]

        if anchor_x is None:
            anchor_x = 10 ** (
                np.log10(xmin)
                + 0.55 * (
                    np.log10(xmax)
                    - np.log10(xmin)
                )
            )

        if anchor_y is None:
            anchor_y = 10 ** (
                np.log10(ymin)
                + 0.7 * (
                    np.log10(ymax)
                    - np.log10(ymin)
                )
            )

        x1 = anchor_x
        x2 = x1 * 10**slope_length_decades

        for i, slope in enumerate(panel_slopes):

            y1 = anchor_y / (3**i)
            y2 = y1 * (x2 / x1) ** slope

            ax.loglog(
                [x1, x2],
                [y1, y2],
                color="0.35",
                linestyle="-",
                linewidth=1,
            )

            ax.text(
                x2 * 1.05,
                y2,
                rf"$k^{{{slope:g}}}$",
                color="0.35",
                va="center",
                fontsize=10,
            )
            
    fig.supxlabel(
        r"$k_\mathrm{r}$ [cpkm]",
        fontsize=12,
        y=0.04,
        fontweight='bold'
    )
    fig.tight_layout()

    return fig, axes

def add_spectral_slope(
    ax,
    slope,
    k_ref,
    y_ref,
    k_min,
    k_max,
    label_k,
    label=None,
    label_factor=1.15,
    fontsize=11
):
    """
    Add a reference spectral slope to an existing log-log axis.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on.

    slope : float
        Spectral slope exponent. For k^-3, use slope=-3.

    k_ref : float
        Reference wavenumber where the spectral value is specified.

    y_ref : float
        Spectral value at k_ref.

    k_min, k_max : float
        Wavenumber range over which to draw the slope line.

    label_k : float
        Wavenumber where the text label should be placed.

    label : str, optional
        Text label. If None, automatically uses something like r'$k^{-3}$'.

    label_factor : float
        Multiplicative vertical offset of label from the slope line.
    """

    # Wavenumbers for reference line
    k = np.logspace(
        np.log10(k_min),
        np.log10(k_max),
        100
    )

    # Power-law line anchored at (k_ref, y_ref)
    y = y_ref * (k / k_ref)**slope

    # Plot slope
    ax.plot(
        k,
        y,
        color='black',
        linestyle=':',
        linewidth=1,
        zorder=3,
    )

    # Value of slope line at label location
    label_y = y_ref * (label_k / k_ref)**slope

    # Automatic label
    if label is None:
        label = rf'$k^{{{slope:g}}}$'

    ax.text(
        label_k,
        label_y * label_factor,
        label,
        fontsize=fontsize,
        color='black',
        ha='center',
        va='bottom',
    )

def plot_spectra(
    spectra,
    title=None,
    slopes=None,
    figsize=(4, 6),
    xlim=None,
    ylim=None,
    slope_length_decades=0.5,
    slope_anchor_x=None,
    slope_anchor_y=None,
    ylab='Spectral Density',
    dpi=100
):
    """
    Plot one or more isotropic spectra on log-log axes.

    Parameters
    ----------
    spectra : dict
        Dictionary of plotting instructions.

        `data` may be either:
            - an xr.DataArray containing the spectrum, or
            - an xr.Dataset returned by CI(), containing:
                spectra
                CI_low
                CI_high

        Example:

        spectra = {
            "LVWD": {
                "data": iso_lvwd_CI,
                "color": "tab:blue",
                "linestyle": "-",
                "label": "LVWD",
                "linewidth": 2,
                "plot_CI": True,
                "CI_alpha": 0.2,
            },
            "HVWD": {
                "data": iso_hvwd,
                "color": "tab:red",
                "linestyle": "--",
                "label": "HVWD",
                "linewidth": 2,
            },
        }

    title : str or None
        Figure title.

    slopes : list of float or None
        Reference slopes, such as [-3, -3.5, -4].

    figsize : tuple
        Figure size.

    xlim, ylim : tuple or None
        Axis limits.

    slope_length_decades : float
        Horizontal length of slope-reference lines in log10 space.

    slope_anchor_x, slope_anchor_y : float or None
        Starting point for the first slope line.

    ylab : str
        Default y-axis label if no long_name attribute is available.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    first_data = None

    for name, settings in spectra.items():

        data = settings["data"]

        # --------------------------------------------------------
        # Data can either be:
        #   DataArray -> spectrum directly
        #   Dataset   -> output from CI()
        # --------------------------------------------------------

        if isinstance(data, xr.Dataset):
            da = data["spectrum"]
            CI_low = data.get("CI_low")
            CI_high = data.get("CI_high")
        else:
            da = data
            CI_low = None
            CI_high = None

        if first_data is None:
            first_data = da

        kr = da["kr"]

        valid = (
            np.isfinite(kr)
            & np.isfinite(da)
            & (kr > 0)
            & (da > 0)
        )

        kr_plot = kr.where(valid, drop=True)
        da_plot = da.where(valid, drop=True)

        color = settings.get("color")

        # --------------------------------------------------------
        # Optional confidence interval
        # --------------------------------------------------------

        if (
            settings.get("plot_CI", False)
            and CI_low is not None
            and CI_high is not None
        ):

            CI_low_plot = CI_low.where(valid, drop=True)
            CI_high_plot = CI_high.where(valid, drop=True)

            ax.fill_between(
                kr_plot.values,
                CI_low_plot.values,
                CI_high_plot.values,
                color=color,
                alpha=settings.get("CI_alpha", 0.2),
                linewidth=0,
                zorder=7,
            )

        # --------------------------------------------------------
        # Spectrum
        # --------------------------------------------------------

        ax.loglog(
            kr_plot,
            da_plot,
            color=color,
            linestyle=settings.get("linestyle", "-"),
            linewidth=settings.get("linewidth", 1.5),
            label=settings.get("label", name),
            alpha=settings.get("alpha", 1),
            zorder=8,
        )

    ax.set_xlabel(r"k$_\mathrm{r}$ [cpkm]")

    if first_data is not None:
        units = first_data.attrs.get("units", "")
        long_name = first_data.attrs.get("long_name", ylab)

        if units:
            ax.set_ylabel(f"{long_name} [{units}]")
        else:
            ax.set_ylabel(long_name)

    if title is not None:
        ax.set_title(title)

    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.7,
        alpha=0.5,
        color="lightgrey",
    )

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    # ------------------------------------------------------------
    # Format wavenumber ticks as reciprocal spatial scales
    # e.g., 10^-3 -> 1/1000, 10^-2 -> 1/100
    # ------------------------------------------------------------

    def reciprocal_formatter(x, pos):
        if x <= 0:
            return ""

        denominator = 1 / x

        if np.isclose(denominator, round(denominator)):
            denominator = int(round(denominator))

        return rf"$1/{denominator:g}$"

    ax.xaxis.set_major_formatter(FuncFormatter(reciprocal_formatter))

    # Draw first so axis limits are known
    fig.canvas.draw()

    # ------------------------------------------------------------
    # Reference slopes
    # ------------------------------------------------------------

    if slopes:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        if slope_anchor_x is None:
            slope_anchor_x = 10 ** (
                np.log10(xmin)
                + 0.55 * (np.log10(xmax) - np.log10(xmin))
            )

        if slope_anchor_y is None:
            slope_anchor_y = 10 ** (
                np.log10(ymin)
                + 0.7 * (np.log10(ymax) - np.log10(ymin))
            )

        x1 = slope_anchor_x
        x2 = x1 * 10**slope_length_decades

        for i, slope in enumerate(slopes):

            y1 = slope_anchor_y / (3**i)
            y2 = y1 * (x2 / x1) ** slope

            ax.loglog(
                [x1, x2],
                [y1, y2],
                color="0.35",
                linestyle="-",
                linewidth=1,
            )

            ax.text(
                x2 * 1.05,
                y2,
                rf"$k^{{{slope:g}}}$",
                color="0.35",
                va="center",
                fontsize=10,
            )

    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=1.0,       # fully opaque
        facecolor="white",    # solid white background
        edgecolor="black",    # solid border
    )

    return fig, ax

def plot_coherence(
    curves,
    coherence_var="coherence_iso",
    cl_var="coherence_CL",
    figsize=(8, 5),
    title=None,
    linewidth=2,
    xlim=None,
):
    """
    Plot isotropic coherence for any number of datasets.

    Parameters
    ----------
    curves : dict
        Dictionary of curves to plot. Format:

        {
            "Label": {
                "ds": dataset,
                "color": "C0",
                "linestyle": "-",
            },
            ...
        }

        Additional matplotlib line kwargs can also be supplied.

    coherence_var : str
        Name of isotropic coherence variable.

    cl_var : str
        Name of coherence significance threshold.

    figsize : tuple
        Figure size.

    title : str, optional
        Figure title.

    linewidth : float
        Default linewidth.

    xlim : tuple, optional
        Wavenumber limits as (xmin, xmax), in cpkm.

    Returns
    -------
    fig, ax
    """

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(
        figsize=figsize,
    )

    # ------------------------------------------------------
    # Plot curves
    # ------------------------------------------------------

    CL = None

    for label, cfg in curves.items():

        ds = cfg["ds"]

        coherence = ds[coherence_var]

        # Same CL for all curves
        if CL is None:
            CL = float(ds[cl_var])

        # Radial-wavenumber coordinate
        kr_dim = coherence.dims[0]
        kr = coherence[kr_dim]

        # --------------------------------------------------
        # Style
        # --------------------------------------------------

        style = {
            key: value
            for key, value in cfg.items()
            if key != "ds"
        }

        style.setdefault(
            "linewidth",
            linewidth,
        )

        # --------------------------------------------------
        # Coherence
        # --------------------------------------------------

        ax.plot(
            kr,
            coherence,
            label=label,
            **style,
        )

    # ------------------------------------------------------
    # Confidence level
    # ------------------------------------------------------

    ax.axhline(
        CL,
        color="k",
        linestyle="--",
        linewidth=1,
    )

    ax.text(
        0.29,
        CL + 0.005,
        f"CL = {CL:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=11,
    )

    # ------------------------------------------------------
    # Wavenumber formatter
    # ------------------------------------------------------

    def wavenumber_formatter(k, pos):

        if k <= 0:
            return ""

        scale_km = 1 / k

        return f"1/{scale_km:g}"

    # ------------------------------------------------------
    # Formatting
    # ------------------------------------------------------

    ax.set_xscale("log")
    ax.set_ylim(0, 1)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.xaxis.set_major_formatter(
        FuncFormatter(wavenumber_formatter)
    )

    ax.set_xlabel(
        r"$k_r$ [cpkm]", fontsize=13
    )

    ax.set_ylabel(
        "Coherence [-]", fontsize=13
    )

    ax.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.35,
    )

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
    )

    if title is not None:
        ax.set_title(title)

    return fig, ax

def plot_variance_lost(
    datasets,
    titles,
    styles,
    variable="cumulative_loss_percent",
    xdim="kr",
    figsize=(8, 6),
    xlim=(1e-3, 1/25),
    ylim=(-10, 70),
    xlabel=r"k$_r$ [cpkm]",
    ylabel="Variance lost [%]",
    legend=True,
):
    """
    Plot variance loss for multiple datasets side-by-side.

    Parameters
    ----------
    datasets : list of xr.Dataset
        Datasets containing variance-loss results.

    titles : list of str
        Panel titles corresponding to each dataset.

    styles : dict
        Plotting styles keyed by experiment name.

    variable : str, default="cumulative_loss_percent"
        Dataset variable to plot.

    xdim : str, default="kr"
        Wavenumber dimension.

    figsize : tuple, default=(8, 6)
        Figure size.

    xlim : tuple, default=(1e-3, 1/25)
        X-axis limits.

    ylim : tuple, default=(-10, 70)
        Y-axis limits.

    xlabel : str
        Shared x-axis label.

    ylabel : str
        Shared y-axis label.

    wavelength_formatter : callable, optional
        Formatter function for the x-axis.

    legend : bool, default=True
        Whether to show the legend on the first panel.

    Returns
    -------
    fig, axes
    """

    n = len(datasets)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    # Ensure axes is iterable if n == 1
    if n == 1:
        axes = [axes]

    for i, (ax, ds, title) in enumerate(
        zip(axes, datasets, titles)
    ):

        for experiment in ds.experiment.values:

            da = ds[variable].sel(
                experiment=experiment
            )

            ax.plot(
                da[xdim],
                da,
                label=experiment,
                **styles[experiment],
            )

        ax.set_title(
            title,
            fontsize=13,
        )

        ax.text(
            0.98,
            0.98,
            f"({chr(97 + i)})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontweight="bold",
            fontsize=18,
        )

        if legend and i == 0:
            ax.legend(
                loc="upper left",
                frameon=True,
                framealpha=1.0,
                facecolor="white",
                edgecolor="black",
            )

        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.axhline(
            0,
            color="k",
            linewidth=1,
            linestyle="--",
        )

        ax.xaxis.set_major_formatter(
            FuncFormatter(wavenumber_formatter)
        )

        ax.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.35,
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=11,
        )

    fig.subplots_adjust(
        left=0.13,
        bottom=0.13,
        wspace=0.16,
    )

    fig.text(
        0.54,
        0.02,
        xlabel,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    fig.text(
        0.04,
        0.52,
        ylabel,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout(
        rect=[0.04, 0.04, 1, 1]
    )

    return fig, axes

def plot_universal_spectra(
    spectra,
    coord="kr",
    title=None,
    slopes=None,
    figsize=(4, 6),
    xlim=None,
    ylim=None,
    slope_length_decades=0.5,
    slope_anchor_x=None,
    slope_anchor_y=None,
    xlab=None,
    ylab="Spectral Density",
    reciprocal_ticks=None,
    dpi=100,
):
    """
    Plot one or more spectra on log-log axes.

    Parameters
    ----------
    spectra : dict
        Dictionary of plotting instructions.

        `data` may be either:
            - an xr.DataArray containing the spectrum, or
            - an xr.Dataset returned by CI(), containing:
                spectrum
                CI_low
                CI_high

    coord : str
        Name of the spectral coordinate/dimension.
        Examples:
            "kr"   -> spatial wavenumber
            "freq" -> frequency

    title : str or None
        Figure title.

    slopes : list of float or None
        Reference slopes, such as [-3, -3.5, -4].

    figsize : tuple
        Figure size.

    xlim, ylim : tuple or None
        Axis limits.

    slope_length_decades : float
        Horizontal length of slope-reference lines in log10 space.

    slope_anchor_x, slope_anchor_y : float or None
        Starting point for the first slope line.

    xlab : str or None
        X-axis label. If None, inferred from coordinate attributes when possible.

    ylab : str
        Default y-axis label if no long_name attribute is available.

    reciprocal_ticks : bool or None
        Whether to display x ticks as reciprocals
        (e.g. 0.001 -> 1/1000).

        If None:
            True  when coord == "kr"
            False otherwise.
    """

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )

    first_data = None
    first_coord = None

    # ------------------------------------------------------------
    # Default reciprocal behavior
    # ------------------------------------------------------------

    if reciprocal_ticks is None:
        reciprocal_ticks = (coord == "kr")

    # ------------------------------------------------------------
    # Plot spectra
    # ------------------------------------------------------------

    for name, settings in spectra.items():

        data = settings["data"]

        # --------------------------------------------------------
        # Data can either be:
        #   DataArray -> spectrum directly
        #   Dataset   -> output from CI()
        # --------------------------------------------------------

        if isinstance(data, xr.Dataset):

            da = data["spectrum"]
            CI_low = data.get("CI_low")
            CI_high = data.get("CI_high")

        else:

            da = data
            CI_low = None
            CI_high = None

        # Remove any singleton dimensions
        da = da.squeeze(drop=True)

        if coord not in da.coords:
            raise ValueError(
                f"Coordinate '{coord}' not found in spectrum '{name}'. "
                f"Available coordinates: {list(da.coords)}"
            )

        if da.dims != (coord,):
            raise ValueError(
                f"Spectrum '{name}' must be one-dimensional along "
                f"'{coord}', but has dimensions {da.dims}."
            )

        # --------------------------------------------------------
        # Align spectrum and confidence intervals
        # --------------------------------------------------------

        if CI_low is not None and CI_high is not None:

            CI_low = CI_low.squeeze(drop=True)
            CI_high = CI_high.squeeze(drop=True)

            da, CI_low, CI_high = xr.align(
                da,
                CI_low,
                CI_high,
                join="inner",
            )

        # --------------------------------------------------------
        # Compute before boolean indexing
        #
        # Plotting requires the actual values anyway. Doing this
        # here prevents Xarray from trying to use a Dask-backed
        # boolean array with drop=True.
        # --------------------------------------------------------

        da = da.compute()

        if CI_low is not None:
            CI_low = CI_low.compute()

        if CI_high is not None:
            CI_high = CI_high.compute()

        x = da[coord].compute()

        # --------------------------------------------------------
        # Store first spectrum for automatic labels
        # --------------------------------------------------------

        if first_data is None:
            first_data = da
            first_coord = x

        # --------------------------------------------------------
        # Remove invalid values for log plotting
        # --------------------------------------------------------

        valid = (
            np.isfinite(x)
            & np.isfinite(da)
            & (x > 0)
            & (da > 0)
        )

        x_plot = x.where(
            valid,
            drop=True,
        )

        da_plot = da.where(
            valid,
            drop=True,
        )

        color = settings.get("color")

        # --------------------------------------------------------
        # Optional confidence interval
        # --------------------------------------------------------

        if (
            settings.get("plot_CI", False)
            and CI_low is not None
            and CI_high is not None
        ):

            ci_valid = (
                valid
                & np.isfinite(CI_low)
                & np.isfinite(CI_high)
                & (CI_low > 0)
                & (CI_high > 0)
            )

            x_CI_plot = x.where(
                ci_valid,
                drop=True,
            )

            CI_low_plot = CI_low.where(
                ci_valid,
                drop=True,
            )

            CI_high_plot = CI_high.where(
                ci_valid,
                drop=True,
            )

            ax.fill_between(
                x_CI_plot.values,
                CI_low_plot.values,
                CI_high_plot.values,
                color=color,
                alpha=settings.get("CI_alpha", 0.2),
                linewidth=0,
                zorder=7,
            )

        # --------------------------------------------------------
        # Spectrum
        # --------------------------------------------------------

        ax.loglog(
            x_plot.values,
            da_plot.values,
            color=color,
            linestyle=settings.get("linestyle", "-"),
            linewidth=settings.get("linewidth", 1.5),
            label=settings.get("label", name),
            alpha=settings.get("alpha", 1),
            zorder=8,
        )

    # ------------------------------------------------------------
    # Axis labels
    # ------------------------------------------------------------

    if xlab is not None:

        ax.set_xlabel(xlab)

    elif first_coord is not None:

        coord_long_name = first_coord.attrs.get(
            "long_name",
            coord,
        )

        coord_units = first_coord.attrs.get(
            "units",
            "",
        )

        if coord_units:
            ax.set_xlabel(
                f"{coord_long_name} [{coord_units}]"
            )
        else:
            ax.set_xlabel(coord_long_name)

    if first_data is not None:

        units = first_data.attrs.get(
            "units",
            "",
        )

        long_name = first_data.attrs.get(
            "long_name",
            ylab,
        )

        if units:
            ax.set_ylabel(
                f"{long_name} [{units}]"
            )
        else:
            ax.set_ylabel(long_name)

    # ------------------------------------------------------------
    # Title / grid / limits
    # ------------------------------------------------------------

    if title is not None:
        ax.set_title(title)

    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.7,
        alpha=0.5,
        color="lightgrey",
    )

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    # ------------------------------------------------------------
    # Optional reciprocal x-axis ticks
    # ------------------------------------------------------------

    if reciprocal_ticks:

        def reciprocal_formatter(x, pos):

            if x <= 0:
                return ""

            denominator = 1 / x

            if np.isclose(
                denominator,
                round(denominator),
            ):
                denominator = int(
                    round(denominator)
                )

            return rf"$1/{denominator:g}$"

        ax.xaxis.set_major_formatter(
            FuncFormatter(
                reciprocal_formatter
            )
        )

    # Draw first so axis limits are known
    fig.canvas.draw()

    # ------------------------------------------------------------
    # Reference slopes
    # ------------------------------------------------------------

    if slopes:

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        if slope_anchor_x is None:

            slope_anchor_x = 10 ** (
                np.log10(xmin)
                + 0.55
                * (
                    np.log10(xmax)
                    - np.log10(xmin)
                )
            )

        if slope_anchor_y is None:

            slope_anchor_y = 10 ** (
                np.log10(ymin)
                + 0.7
                * (
                    np.log10(ymax)
                    - np.log10(ymin)
                )
            )

        x1 = slope_anchor_x
        x2 = (
            x1
            * 10**slope_length_decades
        )

        for i, slope in enumerate(slopes):

            y1 = (
                slope_anchor_y
                / (3**i)
            )

            y2 = (
                y1
                * (x2 / x1) ** slope
            )

            ax.loglog(
                [x1, x2],
                [y1, y2],
                color="0.35",
                linestyle="-",
                linewidth=1,
            )

            ax.text(
                x2 * 1.05,
                y2,
                rf"$k^{{{slope:g}}}$",
                color="0.35",
                va="center",
                fontsize=10,
            )

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------

    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
    )

    return fig, ax