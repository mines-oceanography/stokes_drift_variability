"""Hard-coded 3+4 panel map layouts for WW3 comparison figures.

The figure always contains:
- one three-panel top row; and
- either two or three four-panel rows below it.

Appearance (colormap and colorbar label) is kept in FIELD_STYLES.
Figure-specific contour levels and ticks are passed separately through ``ranges``.
"""

from __future__ import annotations

from string import ascii_lowercase
from typing import Any, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Circle, FancyArrow, Rectangle

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean


# ---------------------------------------------------------------------
# Fixed appearance only. Data limits and ticks do NOT belong here.
# ---------------------------------------------------------------------

FIELD_STYLES: dict[str, dict[str, Any]] = {
    "wind": {
        "cmap": "cividis",
        "colorbar_label": r"$U_{\mathrm{wnd}}$ [m/s]",
    },
    "current": {
        "cmap": cmocean.cm.deep.reversed(),
        "colorbar_label": r"$U_{\mathrm{cur}}$ [cm/s]",
    },
    "stokes": {
        "cmap": cmocean.cm.speed.reversed(),
        "colorbar_label": r"$U_{\mathrm{s}}$ [cm/s]",
    },
    "langmuir": {
        "cmap": "bone_r",
        "colorbar_label": r"La$_\mathrm{t}$ [-]",
    },
    "mss": {
        "cmap": cmocean.cm.dense.reversed(),
        "colorbar_label": r"MSS [-]",
    },
    "hs": {
        "cmap": "inferno",
        "colorbar_label": r"$H_\mathrm{s}$ [m]",
    },
    "stokes_diff": {
        "cmap": cmocean.cm.balance,
        "colorbar_label": r"$\Delta U_\mathrm{s}$ [cm/s]",
    },
    "stokes_diff_pct": {
        "cmap": cmocean.cm.balance,
        "colorbar_label": r"$\Delta U_\mathrm{s}$ [%]",
    },
    "hs_diff_pct": {
        "cmap": cmocean.cm.balance,
        "colorbar_label": r"$\Delta H_\mathrm{s}$ [%]",
    },
    "T0m1_diff_pct": {
        "cmap": cmocean.cm.balance,
        "colorbar_label": r"$\Delta T_{0\mathrm{m}1}$ [%]",
    },
    "T01": {
        "cmap": cmocean.cm.tempo,
        "colorbar_label": r"T$_{01}$ [s]",
    },
    "T0m1": {
        "cmap": cmocean.cm.tempo,
        "colorbar_label": r"$T_{0m1}$ [s]",
    },
    "vorticity": {
        "cmap": cmocean.cm.curl,
        "colorbar_label": r"$\zeta$/f [-]",
    },
    "divergence": {
        "cmap": cmocean.cm.curl,
        "colorbar_label": r"$\delta$/f [-]",
    },
    "wind_diff": {
        "cmap": cmocean.cm.balance,
        "colorbar_label": r"$\Delta U_\mathrm{wnd}$ [m/s]",
    },
    "wind_diff_pct": {
        "cmap": cmocean.cm.balance,
        "colorbar_label": r"$\Delta U_\mathrm{wnd}$ [%]",
    },
}


# These reproduce the physical colorbar dimensions of the original
# 12 x 9 inch notebook figure. They are converted to figure fractions
# at runtime, so the colorbars do not grow when the figure gets taller.
_COLORBAR_INCHES = {
    "x_offset": 0.036,
    "y_offset": 0.045,
    "box_width": 0.84,
    "box_height": 0.54,
    "x_padding": 0.096,
    "bar_height": 0.09,
}

def subset(ds, timestep, lons, lats):
    return ds.sel(time=timestep, longitude=lons, latitude=lats)

def _format_lon_labels(lons: Sequence[float]) -> list[str]:
    degree = "\N{DEGREE SIGN}"
    labels = []
    for lon in lons:
        wrapped = ((float(lon) + 180.0) % 360.0) - 180.0
        labels.append(f"{abs(wrapped):.0f}{degree}{'W' if wrapped < 0 else 'E'}")
    return labels


def _format_lat_labels(lats: Sequence[float]) -> list[str]:
    degree = "\N{DEGREE SIGN}"
    labels = []
    for lat in lats:
        value = float(lat)
        labels.append(f"{abs(value):.0f}{degree}{'S' if value < 0 else 'N'}")
    return labels


def _add_colorbar(mappable, ax, *, label: str, ticks: Sequence[float]) -> None:
    """Add a panel-anchored colorbar with fixed physical dimensions."""

    fig = ax.figure
    panel_bbox = ax.get_position()

    fig_width = fig.get_figwidth()
    fig_height = fig.get_figheight()

    x_offset = _COLORBAR_INCHES["x_offset"] / fig_width
    y_offset = _COLORBAR_INCHES["y_offset"] / fig_height
    box_width = _COLORBAR_INCHES["box_width"] / fig_width
    box_height = _COLORBAR_INCHES["box_height"] / fig_height
    x_padding = _COLORBAR_INCHES["x_padding"] / fig_width
    bar_height = _COLORBAR_INCHES["bar_height"] / fig_height

    x0_box = panel_bbox.x0 + x_offset
    y0_box = panel_bbox.y0 + y_offset

    background = Rectangle(
        (x0_box, y0_box),
        box_width,
        box_height,
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="none",
        alpha=0.70,
        zorder=8,
        clip_on=False,
    )
    fig.add_artist(background)

    x0_bar = x0_box + x_padding
    bar_width = box_width - 2.0 * x_padding

    # Same vertical placement as the original notebook implementation.
    y_center = y0_box + box_height / 1.2
    y0_bar = y_center - bar_height / 2.0

    cbar_ax = fig.add_axes(
        [x0_bar, y0_bar, bar_width, bar_height],
        zorder=9,
    )
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks(ticks)
    cbar.set_label(label, fontsize=9, labelpad=2)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.tick_params(which="minor", bottom=False, top=False)


def _format_map_axis(
    ax,
    panel_label: str,
    *,
    projection,
    lon_ticks: Sequence[float],
    lat_ticks: Sequence[float],
    extent: Sequence[float] | None,
    show_lon_labels: bool,
    show_lat_labels: bool,
) -> None:
    if extent is not None:
        ax.set_extent(extent, crs=projection)

    ax.coastlines(linewidth=0.5, zorder=4)
    ax.add_feature(
        cfeature.LAND,
        facecolor="tan",
        edgecolor="black",
        linewidth=0.5,
        zorder=2,
    )

    gridlines = ax.gridlines(
        crs=projection,
        draw_labels=False,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
        zorder=3,
    )
    gridlines.xlocator = mticker.FixedLocator(lon_ticks)
    gridlines.ylocator = mticker.FixedLocator(lat_ticks)

    ax.set_xticks(lon_ticks, crs=projection)
    if show_lon_labels:
        ax.set_xticklabels(_format_lon_labels(lon_ticks), rotation=50)
    else:
        ax.tick_params(axis="x", labelbottom=False)

    ax.set_yticks(lat_ticks, crs=projection)
    if show_lat_labels:
        ax.set_yticklabels(_format_lat_labels(lat_ticks))
    else:
        ax.tick_params(axis="y", labelleft=False)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.text(
        0.98,
        0.98,
        f"({panel_label})",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        zorder=20,
    )

def make_four_panel_money_figure(
    four_panel_rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    ranges: Mapping[str, Mapping[str, Any]],
    column_titles: Sequence[str] | None = None,
    field_styles: Mapping[str, Mapping[str, Any]] | None = None,
    projection=None,
    data_crs=None,
    lon_ticks: Sequence[float] = tuple(np.arange(-140, -119, 5)),
    lat_ticks: Sequence[float] = tuple(np.arange(30, 46, 5)),
    extent: Sequence[float] | None = None,
    dpi: int = 500,
    hspace: float = 0,
    wspace: float = 0.15,
):
    """
    Create a figure containing only four-panel rows.

    Each panel dictionary requires:
        - "data"
        - "style"

    Optional panel entries:
        - "range_key"
        - "extent"

    Parameters
    ----------
    four_panel_rows : sequence of rows
        Each row must contain exactly 4 panel dictionaries.

    ranges : mapping
        Dictionary of contour levels/ticks keyed by range name.

    column_titles : sequence of str, optional
        Length-4 sequence of titles to apply only to the top row.
    """

    n_rows = len(four_panel_rows)

    if n_rows not in (1, 2, 3):
        raise ValueError(
            "four_panel_rows must contain exactly one, two, or three rows"
        )

    if any(len(row) != 4 for row in four_panel_rows):
        raise ValueError("each row must contain exactly four panels")

    if column_titles is not None and len(column_titles) != 4:
        raise ValueError("column_titles must contain exactly four titles")

    if projection is None:
        projection = ccrs.PlateCarree()

    if data_crs is None:
        data_crs = ccrs.PlateCarree()

    styles = FIELD_STYLES if field_styles is None else field_styles

    # ---------------------------------------------------------
    # Figure size
    # ---------------------------------------------------------

    if n_rows == 1:
        figsize = (12, 4.0)
    elif n_rows == 2:
        figsize = (12, 7.0)
    else:
        figsize = (12, 10.0)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    # ---------------------------------------------------------
    # One simple grid: n_rows x 4
    # ---------------------------------------------------------

    grid = fig.add_gridspec(
        nrows=n_rows,
        ncols=4,
        hspace=hspace,
        wspace=wspace,
    )

    axes = [
        [
            fig.add_subplot(grid[row, col], projection=projection)
            for col in range(4)
        ]
        for row in range(n_rows)
    ]

    # ---------------------------------------------------------
    # Plot panels
    # ---------------------------------------------------------

    mappables = []
    letter_iter = iter(ascii_lowercase)

    for row_index, (axes_row, panel_row) in enumerate(zip(axes, four_panel_rows)):

        for col_index, (ax, panel) in enumerate(zip(axes_row, panel_row)):

            # Required panel entries
            for key in ("data", "style"):
                if key not in panel:
                    raise KeyError(
                        f"panel at row {row_index}, column {col_index} "
                        f"is missing '{key}'"
                    )

            style_key = panel["style"]
            range_key = panel.get("range_key", style_key)

            try:
                style = styles[style_key]
            except KeyError as exc:
                raise KeyError(f"unknown style '{style_key}'") from exc

            try:
                range_spec = ranges[range_key]
            except KeyError as exc:
                raise KeyError(f"unknown range '{range_key}'") from exc

            levels = np.asarray(range_spec["levels"], dtype=float)
            ticks = np.asarray(range_spec["ticks"], dtype=float)

            mappable = panel["data"].plot.contourf(
                ax=ax,
                transform=data_crs,
                cmap=style["cmap"],
                levels=levels,
                extend=range_spec.get("extend", "both"),
                add_colorbar=False,
                add_labels=False,
            )

            # Longitude labels only on bottom row
            show_lon = row_index == n_rows - 1

            # Latitude labels only on left column
            show_lat = col_index == 0

            _format_map_axis(
                ax,
                next(letter_iter),
                projection=projection,
                lon_ticks=lon_ticks,
                lat_ticks=lat_ticks,
                extent=panel.get("extent", extent),
                show_lon_labels=show_lon,
                show_lat_labels=show_lat,
            )

            # Titles only on the top row
            if row_index == 0 and column_titles is not None:
                ax.set_title(column_titles[col_index])
            else:
                ax.set_title("")

            mappables.append(
                (mappable, ax, style["colorbar_label"], ticks)
            )

    # ---------------------------------------------------------
    # Let Cartopy finalize axes before positioning colorbars
    # ---------------------------------------------------------

    fig.canvas.draw()

    # ---------------------------------------------------------
    # Individual panel colorbars
    # ---------------------------------------------------------

    for mappable, ax, label, ticks in mappables:
        _add_colorbar(
            mappable,
            ax,
            label=label,
            ticks=ticks
        )

    return fig, axes


def make_three_panel_money_figure(
    three_panel_rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    ranges: Mapping[str, Mapping[str, Any]],
    field_styles: Mapping[str, Mapping[str, Any]] | None = None,
    projection=None,
    data_crs=None,
    lon_ticks: Sequence[float] = tuple(np.arange(-140, -119, 5)),
    lat_ticks: Sequence[float] = tuple(np.arange(30, 46, 5)),
    extent: Sequence[float] | None = None,
    dpi: int = 500,
    hspace: float = 0,
    wspace: float = 0.15,
):
    """
    Create a figure containing only three-panel rows.

    Each panel dictionary requires:
        - "data"
        - "title"
        - "style"

    Optional panel entries:
        - "range_key"
        - "extent"
    """

    n_rows = len(three_panel_rows)

    if n_rows not in (1, 2, 3):
        raise ValueError(
            "three_panel_rows must contain exactly one, two, or three rows"
        )

    if any(len(row) != 3 for row in three_panel_rows):
        raise ValueError(
            "each row must contain exactly three panels"
        )

    if projection is None:
        projection = ccrs.PlateCarree()

    if data_crs is None:
        data_crs = ccrs.PlateCarree()

    styles = FIELD_STYLES if field_styles is None else field_styles

    # ---------------------------------------------------------
    # Figure size
    # ---------------------------------------------------------

    if n_rows == 1:
        figsize = (12, 4.5)

    elif n_rows == 2:
        figsize = (12, 8.0)

    else:
        figsize = (12, 11.5)

    fig = plt.figure(
        figsize=figsize,
        dpi=dpi
    )

    # ---------------------------------------------------------
    # Grid: n_rows x 3
    # ---------------------------------------------------------

    grid = fig.add_gridspec(
        nrows=n_rows,
        ncols=3,
        hspace=hspace,
        wspace=wspace,
    )

    axes = [
        [
            fig.add_subplot(
                grid[row, col],
                projection=projection
            )
            for col in range(3)
        ]
        for row in range(n_rows)
    ]

    # ---------------------------------------------------------
    # Plot panels
    # ---------------------------------------------------------

    mappables = []
    letter_iter = iter(ascii_lowercase)

    for row_index, (axes_row, panel_row) in enumerate(
        zip(axes, three_panel_rows)
    ):

        for col_index, (ax, panel) in enumerate(
            zip(axes_row, panel_row)
        ):

            # Required panel entries
            for key in ("data", "title", "style"):
                if key not in panel:
                    raise KeyError(
                        f"panel at row {row_index}, "
                        f"column {col_index} "
                        f"is missing '{key}'"
                    )

            style_key = panel["style"]
            range_key = panel.get(
                "range_key",
                style_key
            )

            try:
                style = styles[style_key]
            except KeyError as exc:
                raise KeyError(
                    f"unknown style '{style_key}'"
                ) from exc

            try:
                range_spec = ranges[range_key]
            except KeyError as exc:
                raise KeyError(
                    f"unknown range '{range_key}'"
                ) from exc

            levels = np.asarray(
                range_spec["levels"],
                dtype=float
            )

            ticks = np.asarray(
                range_spec["ticks"],
                dtype=float
            )

            # -------------------------------------------------
            # Plot
            # -------------------------------------------------

            mappable = panel["data"].plot.contourf(
                ax=ax,
                transform=data_crs,
                cmap=style["cmap"],
                levels=levels,
                extend=range_spec.get("extend", "both"),
                add_colorbar=False,
                add_labels=False,
            )

            # Longitude labels only on bottom row
            show_lon = row_index == n_rows - 1

            # Latitude labels only on left column
            show_lat = col_index == 0

            _format_map_axis(
                ax,
                next(letter_iter),
                projection=projection,
                lon_ticks=lon_ticks,
                lat_ticks=lat_ticks,
                extent=panel.get("extent", extent),
                show_lon_labels=show_lon,
                show_lat_labels=show_lat,
            )

            # Per-panel title
            ax.set_title(panel["title"])

            mappables.append(
                (
                    mappable,
                    ax,
                    style["colorbar_label"],
                    ticks
                )
            )

    # ---------------------------------------------------------
    # Let Cartopy finalize positions before adding colorbars
    # ---------------------------------------------------------

    fig.canvas.draw()

    # ---------------------------------------------------------
    # Individual panel colorbars
    # ---------------------------------------------------------

    for mappable, ax, label, ticks in mappables:

        _add_colorbar(
            mappable,
            ax,
            label=label,
            ticks=ticks
        )

    return fig, axes


def make_money_figure(
    top_panels: Sequence[Mapping[str, Any]],
    four_panel_rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    ranges: Mapping[str, Mapping[str, Any]],
    field_styles: Mapping[str, Mapping[str, Any]] | None = None,
    projection=None,
    data_crs=None,
    lon_ticks: Sequence[float] = tuple(np.arange(-140, -119, 5)),
    lat_ticks: Sequence[float] = tuple(np.arange(30, 46, 5)),
    extent: Sequence[float] | None = None,
    dpi: int = 500,
    force_hspace = 0
):
    """Create the hard-coded top-3 plus either two or three lower-4 layout.

    Panel dictionaries require ``data``, ``title``, and ``style``.
    By default, the panel uses the range entry with the same key as ``style``.
    Supply ``range_key`` when the same style needs different limits.
    """

    if len(top_panels) != 3:
        raise ValueError("top_panels must contain exactly three panels")

    n_four_rows = len(four_panel_rows)

    if n_four_rows not in (1, 2, 3):
        raise ValueError(
            "four_panel_rows must contain exactly one, two, or three rows"
        )
    if any(len(row) != 4 for row in four_panel_rows):
        raise ValueError("each lower row must contain exactly four panels")

    if projection is None:
        projection = ccrs.PlateCarree()
    if data_crs is None:
        data_crs = ccrs.PlateCarree()

    styles = FIELD_STYLES if field_styles is None else field_styles

    # Explicit, hard-coded layouts. No arbitrary-row layout logic.
    if n_four_rows == 1:
        figsize = (12, 9)
        lower_height = 4
        lower_hspace = 0  # irrelevant with only one lower row

    elif n_four_rows == 2:
        figsize = (12, 11.5)
        lower_height = 6
        lower_hspace = 0

    else:
        figsize = (12, 14.5)
        lower_height = 9
        lower_hspace = 0

    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Top row is one block; all four-panel rows are a second block.
    # This gives separate control over the top-to-lower gap and the
    # uniform spacing between lower rows.
    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[4, lower_height],
        hspace=force_hspace,
    )

    top_grid = outer_grid[0].subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1, 1, 1],
        wspace=0.15,
    )

    lower_grid = outer_grid[1].subgridspec(
        nrows=n_four_rows,
        ncols=4,
        width_ratios=[1, 1, 1, 1],
        hspace=lower_hspace,
        wspace=0.15,
    )

    top_axes = [
        fig.add_subplot(top_grid[0, col], projection=projection)
        for col in range(3)
    ]
    lower_axes = [
        [
            fig.add_subplot(lower_grid[row, col], projection=projection)
            for col in range(4)
        ]
        for row in range(n_four_rows)
    ]
    axes = [top_axes, *lower_axes]

    panel_rows = [top_panels, *four_panel_rows]
    mappables: list[tuple[Any, Any, str, Sequence[float]]] = []
    letter_iter = iter(ascii_lowercase)

    for row_index, (axes_row, panel_row) in enumerate(zip(axes, panel_rows)):
        for col_index, (ax, panel) in enumerate(zip(axes_row, panel_row)):
            for key in ("data", "title", "style"):
                if key not in panel:
                    raise KeyError(
                        f"panel at row {row_index}, column {col_index} "
                        f"is missing '{key}'"
                    )

            style_key = panel["style"]
            range_key = panel.get("range_key", style_key)

            try:
                style = styles[style_key]
            except KeyError as exc:
                raise KeyError(f"unknown style '{style_key}'") from exc
            try:
                range_spec = ranges[range_key]
            except KeyError as exc:
                raise KeyError(f"unknown range '{range_key}'") from exc

            levels = np.asarray(range_spec["levels"], dtype=float)
            ticks = np.asarray(range_spec["ticks"], dtype=float)

            mappable = panel["data"].plot.contourf(
                ax=ax,
                transform=data_crs,
                cmap=style["cmap"],
                levels=levels,
                extend=range_spec.get("extend", "both"),
                add_colorbar=False,
                add_labels=False,
            )

            # Top row always has longitude labels.
            # Lower rows have them only on the final row.
            show_lon = row_index == 0 or row_index == len(axes) - 1
            show_lat = col_index == 0

            _format_map_axis(
                ax,
                next(letter_iter),
                projection=projection,
                lon_ticks=lon_ticks,
                lat_ticks=lat_ticks,
                extent=panel.get("extent", extent),
                show_lon_labels=show_lon,
                show_lat_labels=show_lat,
            )
            ax.set_title(panel["title"])

            mappables.append(
                (mappable, ax, style["colorbar_label"], ticks)
            )

    # Let Cartopy finalize every GeoAxes position before placing colorbars.
    fig.canvas.draw()

    for mappable, ax, label, ticks in mappables:
        _add_colorbar(mappable, ax, label=label, ticks=ticks)

    return fig, axes


def add_circle_to_axes(
    ax,
    center_lon: float,
    center_lat: float,
    radius: float,
    *,
    color: str = "white",
    linestyle: str = "--",
    linewidth: float = 2.0,
    data_crs=None,
):
    if data_crs is None:
        data_crs = ccrs.PlateCarree()
    circle = Circle(
        (center_lon, center_lat),
        radius,
        edgecolor=color,
        linestyle=linestyle,
        facecolor="none",
        transform=data_crs,
        linewidth=linewidth,
        zorder=20,
    )
    ax.add_patch(circle)
    return circle


def add_arrow_to_axis(
    ax,
    lat: float,
    lon: float,
    length: float,
    angle: float,
    *,
    color: str = "white",
    linewidth: float = 2.0,
    head_width: float = 0.5,
    head_length: float = 0.5,
    data_crs=None,
):
    if data_crs is None:
        data_crs = ccrs.PlateCarree()
    rise = np.sin(np.deg2rad(angle)) * length
    run = np.cos(np.deg2rad(angle)) * length
    arrow = FancyArrow(
        x=lon,
        y=lat,
        dx=run,
        dy=rise,
        transform=data_crs,
        color=color,
        linewidth=linewidth,
        head_width=head_width,
        head_length=head_length,
        shape="full",
        length_includes_head=True,
        zorder=20,
    )
    ax.add_patch(arrow)
    return arrow


def add_box_to_axis(
    ax,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    *,
    edgecolor: str = "white",
    linewidth: float = 2.0,
    linestyle: str = ":",
    data_crs=None,
):
    if data_crs is None:
        data_crs = ccrs.PlateCarree()
    box = Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        transform=data_crs,
        zorder=20,
    )
    ax.add_patch(box)
    return box


__all__ = [
    "FIELD_STYLES",
    "make_money_figure",
    "add_circle_to_axes",
    "add_arrow_to_axis",
    "add_box_to_axis",
]
