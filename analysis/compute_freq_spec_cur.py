# analysis/compute_freq_spec_cur.py

from pathlib import Path

import numpy as np
import xarray as xr
import xrft


# ---------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw_ww3"
    / "lvwd_cur.nc"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "spectra"
OUTPUT_FILE = OUTPUT_DIR / "freq_spec_cur.zarr"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Analysis region
LON_MIN = 360 - 131
LON_MAX = 360 - 122

LAT_MIN = 32
LAT_MAX = 39


# One-year analysis period
TIME_START = np.datetime64("2020-02-01")
TIME_END = np.datetime64("2021-02-01")  # exclusive


# Number of spatial points processed together by Dask
POINT_CHUNK = 256


def main():

    # -----------------------------------------------------------------
    # Open dataset
    #
    # Do not Dask-chunk yet. This lets us cheaply read only the first
    # time slice when constructing the land mask.
    # -----------------------------------------------------------------

    print(f"Opening: {INPUT_FILE}")

    ds = xr.open_dataset(INPUT_FILE)

    try:

        # -------------------------------------------------------------
        # Select variables
        # -------------------------------------------------------------

        print("Selecting current fields...")

        ds = ds[
            [
                "ucur",
                "vcur",
            ]
        ]

        # -------------------------------------------------------------
        # Select region
        # -------------------------------------------------------------

        print(
            f"Selecting region: "
            f"{LON_MIN}–{LON_MAX}°E, "
            f"{LAT_MIN}–{LAT_MAX}°N"
        )

        ds = ds.sel(
            longitude=slice(LON_MIN, LON_MAX),
            latitude=slice(LAT_MIN, LAT_MAX),
        )

        # -------------------------------------------------------------
        # Select one-year time period
        # -------------------------------------------------------------

        print(
            f"Selecting time period: "
            f"{TIME_START} to {TIME_END} (exclusive)"
        )

        ds = ds.sel(
            time=(
                (ds.time >= TIME_START)
                & (ds.time < TIME_END)
            )
        )

        print(
            f"Selected {ds.sizes['time']} time steps, "
            f"{ds.sizes['latitude']} latitudes, "
            f"{ds.sizes['longitude']} longitudes"
        )

        # -------------------------------------------------------------
        # Identify ocean points
        #
        # NaNs are assumed to represent the static land mask, so only
        # the first time slice is inspected.
        # -------------------------------------------------------------

        print(
            "Identifying ocean grid points from the first time slice..."
        )

        valid_2d = (
            ds["ucur"].isel(time=0).notnull()
            & ds["vcur"].isel(time=0).notnull()
        ).load()

        n_total = (
            ds.sizes["latitude"]
            * ds.sizes["longitude"]
        )

        n_valid = int(
            valid_2d.sum().item()
        )

        print(
            f"Found {n_valid} ocean grid points "
            f"out of {n_total} total points"
        )

        # -------------------------------------------------------------
        # Stack spatial dimensions
        # -------------------------------------------------------------

        print(
            "Stacking latitude and longitude into a point dimension..."
        )

        ds = ds.stack(
            point=("latitude", "longitude")
        )

        valid = valid_2d.stack(
            point=("latitude", "longitude")
        )

        # Pass only the boolean values to avoid carrying the scalar
        # time coordinate from the first-time-slice mask into isel().
        ds = ds.isel(
            point=valid.values
        )

        print(
            f"Using {ds.sizes['point']} ocean grid points"
        )

        # -------------------------------------------------------------
        # Replace the stacked MultiIndex with an ordinary point index
        #
        # latitude and longitude remain coordinates along point.
        # This makes the result much cleaner to save to Zarr.
        # -------------------------------------------------------------

        print("Preparing spatial coordinates...")

        ds = ds.reset_index("point")

        ds = ds.assign_coords(
            point=np.arange(
                ds.sizes["point"]
            )
        )

        # -------------------------------------------------------------
        # Convert time to elapsed hours
        #
        # xrft determines the frequency units from the coordinate
        # spacing. Using hours gives frequency initially in cycles/hour.
        # -------------------------------------------------------------

        print("Converting time coordinate to elapsed hours...")

        time_hours = (
            (ds.time - ds.time.isel(time=0))
            / np.timedelta64(1, "h")
        )

        ds = ds.assign_coords(
            time=time_hours
        )

        ds["time"].attrs["units"] = "hours"

        # -------------------------------------------------------------
        # Dask chunking
        #
        # xrft requires the transformed dimension to be one chunk.
        # Therefore the entire time dimension is one chunk, while the
        # point dimension is divided into smaller independent chunks.
        # -------------------------------------------------------------

        print(
            f"Chunking data with {POINT_CHUNK} spatial points per chunk..."
        )

        ds = ds.chunk(
            {
                "time": -1,
                "point": POINT_CHUNK,
            }
        )

        print(
            f"Chunks: {ds.chunks}"
        )

        # -------------------------------------------------------------
        # u-current frequency spectrum
        # -------------------------------------------------------------

        print("Computing u-current frequency spectra...")

        Su = xrft.power_spectrum(
            ds["ucur"],
            dim="time",
            detrend="linear",
            window="hann",
            scaling="density",
            window_correction=True,
        )

        # -------------------------------------------------------------
        # v-current frequency spectrum
        # -------------------------------------------------------------

        print("Computing v-current frequency spectra...")

        Sv = xrft.power_spectrum(
            ds["vcur"],
            dim="time",
            detrend="linear",
            window="hann",
            scaling="density",
            window_correction=True,
        )

        # -------------------------------------------------------------
        # Positive frequencies only
        # -------------------------------------------------------------

        print("Keeping positive frequencies only...")

        Su = 2 * Su.where(
            Su["freq_time"] > 0,
            drop=True,
        )

        Sv = 2 * Sv.where(
            Sv["freq_time"] > 0,
            drop=True,
        )

        # -------------------------------------------------------------
        # Convert cycles/hour -> cycles/day
        #
        # Frequency:
        #
        #     f_cpd = 24 f_cph
        #
        # Because scaling="density", the spectral density must also
        # transform so that integrated variance is unchanged:
        #
        #     S_cpd = S_cph / 24
        # -------------------------------------------------------------

        print(
            "Converting frequency coordinate to cycles per day..."
        )

        Su = Su.rename(
            {"freq_time": "freq"}
        )

        Sv = Sv.rename(
            {"freq_time": "freq"}
        )

        Su = Su.assign_coords(
            freq=Su["freq"] * 24
        )

        Sv = Sv.assign_coords(
            freq=Sv["freq"] * 24
        )

        Su = Su / 24
        Sv = Sv / 24

        Su["freq"].attrs = {
            "long_name": "frequency",
            "units": "cpd",
        }

        Sv["freq"].attrs = {
            "long_name": "frequency",
            "units": "cpd",
        }

        # -------------------------------------------------------------
        # Vector variance spectrum
        #
        # S_vector = S_u + S_v
        #
        # This is vector variance, not kinetic energy, so there is no
        # factor of 1/2.
        # -------------------------------------------------------------

        print("Computing vector variance spectra...")

        vector_variance_spectrum = (
            Su + Sv
        )

        # -------------------------------------------------------------
        # Name variables
        # -------------------------------------------------------------

        Su.name = "u_variance_spectrum"
        Sv.name = "v_variance_spectrum"

        vector_variance_spectrum.name = (
            "vector_variance_spectrum"
        )

        # -------------------------------------------------------------
        # Variable metadata
        # -------------------------------------------------------------

        Su.attrs = {
            "long_name": (
                "frequency spectrum of eastward current variance"
            ),
            "units": "(m s^-1)^2 cpd^-1",
        }

        Sv.attrs = {
            "long_name": (
                "frequency spectrum of northward current variance"
            ),
            "units": "(m s^-1)^2 cpd^-1",
        }

        vector_variance_spectrum.attrs = {
            "long_name": (
                "frequency spectrum of horizontal current "
                "vector variance"
            ),
            "definition": "S_u + S_v",
            "units": "(m s^-1)^2 cpd^-1",
        }

        # -------------------------------------------------------------
        # Assemble output
        #
        # Spectra remain at each individual grid point. Spatial
        # averaging can therefore be done later in notebooks.
        # -------------------------------------------------------------

        print("Assembling output dataset...")

        out = xr.Dataset(
            {
                "u_variance_spectrum": Su,
                "v_variance_spectrum": Sv,
                "vector_variance_spectrum": (
                    vector_variance_spectrum
                ),
            }
        )

        out.attrs = {
            "description": (
                "Temporal frequency spectra of horizontal currents "
                "at individual ocean grid points"
            ),
            "experiment": "lvwd_cur",
            "time_start": str(TIME_START),
            "time_end_exclusive": str(TIME_END),
            "longitude_min": LON_MIN,
            "longitude_max": LON_MAX,
            "latitude_min": LAT_MIN,
            "latitude_max": LAT_MAX,
            "n_valid_points": n_valid,
            "detrend": "linear",
            "window": "hann",
            "scaling": "density",
            "window_correction": True,
        }

        print(
            f"Output dimensions: "
            f"{out.sizes['point']} points × "
            f"{out.sizes['freq']} frequencies"
        )

        # -------------------------------------------------------------
        # Save
        #
        # This is where the actual FFT calculations are triggered.
        # -------------------------------------------------------------

        print(
            f"Computing spectra and saving to:\n"
            f"{OUTPUT_FILE}"
        )

        out.to_zarr(
            OUTPUT_FILE,
            mode="w",
        )

        print("Finished successfully.")

    finally:

        print("Closing input dataset...")
        ds.close()


if __name__ == "__main__":
    main()