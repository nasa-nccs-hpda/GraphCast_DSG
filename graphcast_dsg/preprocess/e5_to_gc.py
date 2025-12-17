# This is the driver code for converting ERA5 data to GraphCast format

import os
import argparse
import logging
import numpy as np
import xesmf as xe
import xarray as xr
import pandas as pd

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

    # add datetime coordinate
    ds = ds.assign_coords(datetime=ds["time"])

    # expand the dimensions 
    for var in ds.data_vars:
        if 'time' in ds[var].dims:
            ds[var] = ds[var].expand_dims("batch")
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
    start_shift = start - pd.Timedelta(hours=12)
    dates = pd.date_range(start=start_shift, end=end, freq="12h")
    return dates

def run_preprocess(
            start_date: str,
            end_date: str,
            outdir: str,
            res_value: float = 0.25,  # 0.25 resolution
            nsteps: int = 40,  # 10 day rollout
        ):

    os.makedirs(outdir, exist_ok=True)

    dates = generate_dates(start_date, end_date)
    pairs = [(dates[i], dates[i+1]) for i in range(len(dates)-1)]

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
        vlist, lvs = get_vars_list_and_levels()
        ds_gs = xr.open_dataset(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        engine="zarr",
        storage_options={"token": None}  # Public dataset, so no authentication needed
        )[vlist].sel(time=list(time_tuple), level=lvs).isel(latitude=slice(None, None, -1))
        
        # convert to graphcast format
        ds_out = to_graphcast_inpout(ds_gs)

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