# scripts/compute_current_statistics.py

from pathlib import Path
import sys

import xarray as xr

# Allow imports from ../src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cur_kinematics import compute_for_ds



def main():

    project_root = Path(__file__).resolve().parents[1]

    input_file = project_root / "data" / "raw_ww3" / "lvwd_cur.nc"
    output_dir = project_root / "data" / "aux"
    output_file = output_dir / "cur_kinematics.nc"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Open WW3 output
    ds = xr.open_dataset(input_file)

    # Compute current statistics
    stats = compute_for_ds(
        ds,
        u_name="ucur",
        v_name="vcur",
        lat_name="lat",
        lon_name="lon",
    )

    # Save
    stats.to_netcdf(output_file)

    print(f"Saved: {output_file}")



if __name__ == "__main__":
    main()