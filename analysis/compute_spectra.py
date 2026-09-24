# analysis/compute_spectra.py

from pathlib import Path
import sys
import gc

import xarray as xr


# ---------------------------------------------------------------------
# Import project functions from ../src
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spectral_analysis import compute_spectra


# ---------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------

INPUT_DIR = PROJECT_ROOT / "data" / "regridded"
OUTPUT_DIR = PROJECT_ROOT / "data" / "spectra"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_START = "2020-02-01"
TIME_END = "2021-03-01"  # exclusive

TIME_BLOCK = 168

# Compute the lowest-level 2-D variance spectrum of each component.
SPECTRAL_VARIABLES = [
    "hs",
    "uuss",
    "vuss",
    "uwnd",
    "vwnd",
    "ucur",
    "vcur",
]


def main():

    input_paths = sorted(INPUT_DIR.glob("*.zarr"))

    for input_path in input_paths:

        output_path = OUTPUT_DIR / input_path.name

        print(f"\nWorking on {input_path.name}")

        ds = xr.open_zarr(input_path, chunks={})

        # Restrict all experiments to the common analysis period.
        ds = ds.sel(
            time=(ds.time >= TIME_START) & (ds.time < TIME_END)
        )

        # Only use variables present in this particular experiment.
        variables = [
            var for var in SPECTRAL_VARIABLES
            if var in ds.data_vars
        ]

        print(f"  Variables: {', '.join(variables)}")
        print(
            f"  Time: {ds.time.values[0]} to "
            f"{ds.time.values[-1]}"
        )

        ds = ds[variables]

        nt = ds.sizes["time"]
        first_block = True

        for start in range(0, nt, TIME_BLOCK):

            stop = min(start + TIME_BLOCK, nt)

            print(f"  Time {start}:{stop}")

            ds_block = ds.isel(time=slice(start, stop))

            spectra = compute_spectra(ds_block).astype("float32")

            if first_block:
                spectra.to_zarr(
                    output_path,
                    mode="w",
                    consolidated=False,
                )
                first_block = False

            else:
                spectra.to_zarr(
                    output_path,
                    mode="a",
                    append_dim="time",
                    consolidated=False,
                )

            del ds_block, spectra
            gc.collect()

        ds.close()

        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()