# analysis/compute_coherence_currents.py

from pathlib import Path
import sys

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------
# Import project functions from ../src
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coherence import scalar_coherence_spectrum


# ---------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------

INPUT_DIR = PROJECT_ROOT / "data" / "regridded"
OUTPUT_DIR = PROJECT_ROOT / "data" / "coherence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_START = np.datetime64("2020-02-01")
TIME_END = np.datetime64("2021-02-01")  # exclusive

DECORRELATION_HOURS = 24 * 7


def open_run(name):
    """Open a regridded WW3 run over the common analysis period."""

    ds = xr.open_zarr(
        INPUT_DIR / f"{name}.zarr",
        chunks={},
        consolidated=False,
    )

    return ds.sel(
        time=(ds.time >= TIME_START) & (ds.time < TIME_END)
    )


def vector_magnitude(u, v):
    """Return the magnitude of a 2-D vector field."""

    return np.hypot(u, v)


def vorticity(u, v):
    """
    Vertical relative vorticity:

        zeta = dv/dx - du/dy

    x and y coordinates are assumed to be in km.
    Output units are s^-1.
    """

    # differentiate() gives derivative per km
    dvdx = v.differentiate("x")
    dudy = u.differentiate("y")

    # Convert from (m/s)/km -> (m/s)/m = s^-1
    vort = (dvdx - dudy) / 1000.0

    vort.name = "vorticity"
    vort.attrs["units"] = "s^-1"

    return vort


def divergence(u, v):
    """
    Horizontal divergence:

        div = du/dx + dv/dy

    x and y coordinates are assumed to be in km.
    Output units are s^-1.
    """

    # differentiate() gives derivative per km
    dudx = u.differentiate("x")
    dvdy = v.differentiate("y")

    # Convert from (m/s)/km -> (m/s)/m = s^-1
    div = (dudx + dvdy) / 1000.0

    div.name = "divergence"
    div.attrs["units"] = "s^-1"

    return div


def main():

    # -----------------------------------------------------------------
    # Open WW3 experiments
    # -----------------------------------------------------------------

    lvwd = open_run("lvwd")
    lvwd_cur = open_run("lvwd_cur")
    noref = open_run("noref")
    nocrt = open_run("nocrt")

    try:

        # -------------------------------------------------------------
        # Stokes-drift magnitudes
        # -------------------------------------------------------------

        lvwd_us = vector_magnitude(
            lvwd["uuss"],
            lvwd["vuss"],
        )

        lvwd_cur_us = vector_magnitude(
            lvwd_cur["uuss"],
            lvwd_cur["vuss"],
        )

        noref_us = vector_magnitude(
            noref["uuss"],
            noref["vuss"],
        )

        nocrt_us = vector_magnitude(
            nocrt["uuss"],
            nocrt["vuss"],
        )

        # -------------------------------------------------------------
        # Current-induced Stokes-drift perturbations
        # -------------------------------------------------------------

        us_lvwd_cur = lvwd_cur_us - lvwd_us
        us_noref = noref_us - lvwd_us
        us_nocrt = nocrt_us - lvwd_us

        # -------------------------------------------------------------
        # Current vorticity and divergence
        # -------------------------------------------------------------

        print("Computing current vorticity and divergence...")

        vort = vorticity(
            lvwd_cur["ucur"],
            lvwd_cur["vcur"],
        )

        div = divergence(
            lvwd_cur["ucur"],
            lvwd_cur["vcur"],
        )

        # -------------------------------------------------------------
        # Current-kinematics–Stokes coherence
        # -------------------------------------------------------------

        print("Computing divergence coherence...")

        results = {
            "div_us_lvwd_cur": scalar_coherence_spectrum(
                div,
                us_lvwd_cur,
                decorrelation_hours=DECORRELATION_HOURS,
            ),

            "div_us_noref": scalar_coherence_spectrum(
                div,
                us_noref,
                decorrelation_hours=DECORRELATION_HOURS,
            ),

            "div_us_nocrt": scalar_coherence_spectrum(
                div,
                us_nocrt,
                decorrelation_hours=DECORRELATION_HOURS,
            ),
        }

        print("Computing vorticity coherence...")

        results.update(
            {
                "vort_us_lvwd_cur": scalar_coherence_spectrum(
                    vort,
                    us_lvwd_cur,
                    decorrelation_hours=DECORRELATION_HOURS,
                ),

                "vort_us_noref": scalar_coherence_spectrum(
                    vort,
                    us_noref,
                    decorrelation_hours=DECORRELATION_HOURS,
                ),

                "vort_us_nocrt": scalar_coherence_spectrum(
                    vort,
                    us_nocrt,
                    decorrelation_hours=DECORRELATION_HOURS,
                ),
            }
        )

        # -------------------------------------------------------------
        # Save
        # -------------------------------------------------------------

        for name, result in results.items():

            output_path = OUTPUT_DIR / f"{name}.zarr"

            print(f"Saving {output_path}")

            result.to_zarr(
                output_path,
                mode="w",
            )

    finally:

        lvwd.close()
        lvwd_cur.close()
        noref.close()
        nocrt.close()


if __name__ == "__main__":
    main()