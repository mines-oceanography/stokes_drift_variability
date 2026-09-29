# analysis/compute_coherence_winds.py

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
TIME_END = np.datetime64("2021-03-01")  # exclusive

DECORRELATION_HOURS = 24 * 7


def open_run(name):
    """Open a regridded WW3 run over the common analysis period."""

    ds = xr.open_zarr(
        INPUT_DIR / f"{name}.zarr",
        chunks={},
    )

    return ds.sel(
        time=(ds.time >= TIME_START) & (ds.time < TIME_END)
    )


def vector_magnitude(u, v):
    """Return the magnitude of a 2-D vector field."""

    return np.hypot(u, v)


def main():

    # -----------------------------------------------------------------
    # Open WW3 experiments
    # -----------------------------------------------------------------

    lvwd = open_run("lvwd")
    hvwd = open_run("hvwd")
    lvwd_cur = open_run("lvwd_cur")
    hvwd_cur = open_run("hvwd_cur")

    try:

        # -------------------------------------------------------------
        # Wind magnitudes
        # -------------------------------------------------------------

        lvwd_wind = vector_magnitude(
            lvwd["uwnd"],
            lvwd["vwnd"],
        )

        hvwd_wind = vector_magnitude(
            hvwd["uwnd"],
            hvwd["vwnd"],
        )

        # -------------------------------------------------------------
        # Stokes-drift magnitudes
        # -------------------------------------------------------------

        lvwd_us = vector_magnitude(
            lvwd["uuss"],
            lvwd["vuss"],
        )

        hvwd_us = vector_magnitude(
            hvwd["uuss"],
            hvwd["vuss"],
        )

        lvwd_cur_us = vector_magnitude(
            lvwd_cur["uuss"],
            lvwd_cur["vuss"],
        )

        hvwd_cur_us = vector_magnitude(
            hvwd_cur["uuss"],
            hvwd_cur["vuss"],
        )

        # -------------------------------------------------------------
        # Wind–Stokes coherence
        # -------------------------------------------------------------

        results = {
            "lvwd_us": scalar_coherence_spectrum(
                lvwd_wind,
                lvwd_us,
                decorrelation_hours=DECORRELATION_HOURS,
            ),

            "hvwd_us": scalar_coherence_spectrum(
                hvwd_wind,
                hvwd_us,
                decorrelation_hours=DECORRELATION_HOURS,
            ),

            "lvwd_cur_us": scalar_coherence_spectrum(
                lvwd_wind,
                lvwd_cur_us,
                decorrelation_hours=DECORRELATION_HOURS,
            ),

            "hvwd_cur_us": scalar_coherence_spectrum(
                hvwd_wind,
                hvwd_cur_us,
                decorrelation_hours=DECORRELATION_HOURS,
            ),
        }

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
        hvwd.close()
        lvwd_cur.close()
        hvwd_cur.close()


if __name__ == "__main__":
    main()