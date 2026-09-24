# scripts/regrid_ww3_output.py

from pathlib import Path
import sys
import gc

import xarray as xr
from dask.distributed import Client, LocalCluster


# ---------------------------------------------------------------------
# Import project functions from ../src
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regrid import regrid_dataset_to_zarr


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_DIR = PROJECT_ROOT / "data" / "raw_ww3"
OUTPUT_DIR = PROJECT_ROOT / "data" / "regridded"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_DIR = Path.home() / "tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Processing settings
# ---------------------------------------------------------------------

# This is the spectral analysis domain, transformed into the longitude
# convention used by WW3
LON_BOUNDS = (-131 + 360, -122 + 360)
LAT_BOUNDS = (32, 39)

# These are the final bounds of the output dataset. While the data in
# this domain goes from 0 to 700 km, we crop at 698 km to avoid a hand-
# full of NaN land grid cells.
X_BOUNDS = (0, 698)
Y_BOUNDS = (0, 698)

# Set this to a number that makes sense for your computing resources
BATCH_SIZE = 8

# ---------------------------------------------------------------------
# WW3 experiments
# ---------------------------------------------------------------------

RUNS = {
    "lvwd": {
        "input": RAW_DIR / "lvwd.nc",
        "variables": ["uwnd", "vwnd", "hs", "uuss", "vuss"],
        "output": OUTPUT_DIR / "lvwd.zarr",
    },

    "hvwd": {
        "input": RAW_DIR / "hvwd.nc",
        "variables": ["uwnd", "vwnd", "hs", "uuss", "vuss"],
        "output": OUTPUT_DIR / "hvwd.zarr",
    },
    # The wind forcing is the same as LVWD, so we don't output
    # it here to avoid large duplicate data. The same idea applies
    # to the high-variability winds and currents (only regrid them
    # in 1 dataset).
    "lvwd_cur": {
        "input": RAW_DIR / "lvwd_cur.nc",
        "variables": ["ucur", "vcur", "hs", "uuss", "vuss"],
        "output": OUTPUT_DIR / "lvwd_cur.zarr",
    },

    "hvwd_cur": {
        "input": RAW_DIR / "hvwd_cur.nc",
        "variables": ["hs", "uuss", "vuss"],
        "output": OUTPUT_DIR / "hvwd_cur.zarr",
    },

    "noref": {
        "input": RAW_DIR / "noref.nc",
        "variables": ["hs", "uuss", "vuss"],
        "output": OUTPUT_DIR / "noref.zarr",
    },

    "nocrt": {
        "input": RAW_DIR / "nocrt.nc",
        "variables": ["hs", "uuss", "vuss"],
        "output": OUTPUT_DIR / "nocrt.zarr",
    },

    "norel": {
        "input": RAW_DIR / "norel.nc",
        "variables": ["hs", "uuss", "vuss"],
        "output": OUTPUT_DIR / "norel.zarr",
    },

    "noadv": {
        "input": RAW_DIR / "noadv.nc",
        "variables": ["hs", "uuss", "vuss"],
        "output": OUTPUT_DIR / "noadv.zarr",
    },
}


def main():

    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        memory_limit="4GiB",
        local_directory=TEMP_DIR,
    )

    client = Client(cluster)

    try:
        for scenario, config in RUNS.items():

            print(f"\nProcessing {scenario}")
            print(f"Input:  {config['input']}")
            print(f"Output: {config['output']}")

            with xr.open_dataset(
                config["input"],
                cache=False,
            ) as raw:

                ds = raw[config["variables"]].sel(
                    longitude=slice(*LON_BOUNDS),
                    latitude=slice(*LAT_BOUNDS),
                )

                regrid_dataset_to_zarr(
                    ds=ds,
                    output_path=config["output"],
                    client=client,
                    batch_size=BATCH_SIZE,
                    x_bounds=X_BOUNDS,
                    y_bounds=Y_BOUNDS,
                    overwrite=False,
                    show_progress=True,
                    source_file=config["input"],
                    scenario=scenario,
                )

            del ds
            gc.collect()

    finally:
        client.close()
        cluster.close()
        gc.collect()


if __name__ == "__main__":
    main()