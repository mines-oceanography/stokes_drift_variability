import xarray as xr

def subset(ds, timestep, lons, lats):
    return ds.sel(time=timestep, longitude=lons, latitude=lats)