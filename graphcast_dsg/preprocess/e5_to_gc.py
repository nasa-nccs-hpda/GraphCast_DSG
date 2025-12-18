# This is the driver code for converting ERA5 data to GraphCast format

import os
import argparse
import logging
import numpy as np
import xarray as xr
import pandas as pd

# Disable Google Cloud authentication
os.environ["NO_GCE_CHECK"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = ""
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

def expand_time_dims(ds, steps):
    # Expand the time dimension of the dataset
    orig_time = ds.time.values
    extra_steps = steps

    ds_last = ds.isel(time=1)
    new_times = pd.date_range(start=orig_time[-1], periods=extra_steps+1, freq="6h")[1:]

    repeated = ds_last.expand_dims(time=range(extra_steps)).copy(deep=True)
    repeated['time'] = new_times
    ds_extended = xr.concat([ds, repeated], dim='time')
    return ds_extended

def to_graphcast_inpout(ds: xr.Dataset) -> xr.Dataset:
    # modify the dataset to match GraphCast expected format

    # change dimension names
    ds = ds.rename({
        "latitude": "lat",
        "longitude": "lon",
    })

    # rename the precipitation variable
    ds = ds.rename({
        "total_precipitation": "total_precipitation_6hr",
    })

    # expand time dimension to 40 steps
    ds = expand_time_dims(ds, steps=40)

    # add datetime coordinate
    ds = ds.assign_coords(datetime=ds["time"])

    # add the dimensions "batch"
    for var in list(ds.data_vars)+['datetime']:
            ds[var] = ds[var].expand_dims("batch")
    
    # drop the time dimension for land_sea_mask and geopotential_at_surface
    ds['land_sea_mask'] = ds['land_sea_mask'].isel(time=0).drop_vars(["time"]).squeeze()
    ds['geopotential_at_surface'] = ds['geopotential_at_surface'].isel(time=0).drop_vars(["time"]).squeeze()  
    return ds

def get_vars_list_and_levels():
    levs = np.array(
    [50,  100,  150,  200,  250,  \
     300,  400,  500,  600,  700, \
     850,  925,  1000])

    static = ["land_sea_mask",
            "geopotential_at_surface",]

    var_2d = ["2m_temperature",
            "toa_incident_solar_radiation",
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation",]

    var_3d = ["temperature",
            "specific_humidity",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "geopotential",]

    var_list = static + var_2d + var_3d
    return var_list, levs

def generate_dates(start_date: str, end_date: str):
    fmt = "%Y-%m-%d:%H"
    try:
        start = pd.to_datetime(start_date, format=fmt)
        end = pd.to_datetime(end_date, format=fmt)
    except:
        raise ValueError(
            "Please provide dates in YYYY-MM-DD:HH (e.g. '2020-01-01:00')")
    start_shift = start - pd.Timedelta(hours=6)
    dates = pd.date_range(start=start_shift, end=end, freq="6h")
    return dates

def run_preprocess(
            start_date: str,
            end_date: str,
            outdir: str,
            nsteps: int = 40,  # 10 day rollout
        ):
    res_value = 0.25  # fixed for ERA5 0.25 deg
    
    os.makedirs(outdir, exist_ok=True)

    dates = generate_dates(start_date, end_date)
    pairs = [(dates[i], dates[i+1]) for i in range(len(dates)-1)]

    vlist, lvs = get_vars_list_and_levels()
    ds_gs = xr.open_dataset(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    engine="zarr",
    storage_options={"token": None,
                        "anon": True,  # Explicitly use anonymous access
                    }  # Public dataset, so no authentication needed
    )[vlist].sel(level=lvs)

    for time_tuple in pairs:

        # create output filename
        date_str = time_tuple[1].strftime("%Y-%m-%dT%H")
        out_file = os.path.join(
            outdir,
            f"graphcast-dataset-source-era5_date-{date_str}"
            + f"_res-{res_value}_levels-13_steps-{nsteps}.nc",
        )

        # skip if already exists
        if os.path.exists(out_file):
            logging.info(f"Skipping {out_file}, already exists.")
            continue
        
        # extract data for the day
        ds_c = ds_gs.sel(time=list(time_tuple)).isel(latitude=slice(None, None, -1))
        
        # convert to graphcast format
        ds_out = to_graphcast_inpout(ds_c)

        # save to netcdf
        ds_out.to_netcdf(
            out_file, mode="w", format="NETCDF4", engine="netcdf4"
        )
    return


def main():

    parser = argparse.ArgumentParser(
        description="Convert GEOS-FP data to ERA5 format."
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date to process (YYYY-MM-DD:HH)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date to process (YYYY-MM-DD:HH)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./output/",
        help="Output directory for the converted files",
    )

    parser.add_argument(
        "--nsteps",
        type=int,
        default=40,
        help="Number of steps for rollout (default 40, 10 days)",
    )

    args = parser.parse_args()

    # Run preprocessing function
    run_preprocess(
        args.start_date,
        args.end_date,
        args.outdir,
        args.nsteps,
    )

    return

if __name__ == "__main__":
    main()