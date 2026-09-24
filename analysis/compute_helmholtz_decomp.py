# analysis/compute_helmholtz_decomp.py

from pathlib import Path
import sys

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------
# Import project functions from ../src
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmholtz import helmholtz_decomposition


# ---------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------

INPUT_FILE = PROJECT_ROOT / "data" / "regridded" / "lvwd_cur.zarr"
OUTPUT_DIR = PROJECT_ROOT / "data" / "spectra"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "helmholtz_decomp.zarr"

TIME_START = np.datetime64("2020-02-01")
TIME_END = np.datetime64("2021-03-01")  # exclusive


def main():

    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    ds = xr.open_zarr(INPUT_FILE, chunks={})

    # Restrict to common analysis period.
    ds = ds.sel(
        time=(ds.time >= TIME_START) & (ds.time < TIME_END)
    )

    print(
        f"Time: {ds.time.values[0]} to "
        f"{ds.time.values[-1]}"
    )

    # Helmholtz decomposition of the current field.
    decomp = helmholtz_decomposition(
        ds["ucur"],
        ds["vcur"],
    )

    decomp.to_zarr(
        OUTPUT_FILE,
        mode="w",
    )

    ds.close()

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()