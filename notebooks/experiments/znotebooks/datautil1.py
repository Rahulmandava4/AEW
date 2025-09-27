import os
from functools import partial
from datetime import timedelta
import numpy as np
import pandas as pd
import xarray as xr

def variable_helper(variable):
    """
    Help with numerous variable attributes.
    """
    var_dict = {
        "sp":   ["SP",   False, "e5.oper.an.sfc"],
        "crr":  ["CCR",  False, "e5.oper.fc.sfc.instan"],
        "lsrr": ["LSRR", False, "e5.oper.fc.sfc.instan"],
        "ishf": ["ISHF", False, "e5.oper.an.sfc"],
        "ie":   ["IE",   False, "e5.oper.an.sfc"],
        "tcw":  ["TCW",  False, "e5.oper.an.sfc"],
        "tcwv": ["TCWV", False, "e5.oper.an.sfc"],
        "t":    ["T",    True,  "e5.oper.an.pl"],
        "u":    ["U",    True,  "e5.oper.an.pl"],
        "v":    ["V",    True,  "e5.oper.an.pl"],
        "q":    ["Q",    True,  "e5.oper.an.pl"],
        "w":    ["W",    True,  "e5.oper.an.pl"],
        "r":    ["R",    True,  "e5.oper.an.pl"],
        "sstk": ["SSTK", False, "e5.oper.an.sfc"],
        "cape": ["CAPE", False, "e5.oper.an.sfc"],
        "pv":   ["PV",   True,  "e5.oper.an.pl"],
        "vo":   ["VO",   True,  "e5.oper.an.pl"],
        "d":    ["D",    True,  "e5.oper.an.pl"],
        "ttr":  ["TTR",  False, "e5.oper.fc.sfc.accumu"],
    }
    return var_dict[variable]

def isValInLst(val, lst):
    """
    Check if item in string in list. Return index.
    """
    return [index for index, content in enumerate(lst) if val in content]

def _initpreprocess(ds, time, plevel_=None):
    """
    Help preprocessing files upon opening by selecting time and pressure level.
    """
    # select time
    ds = ds.sel(time=pd.to_datetime(time))
    
    if plevel_ is not None:
        ds = ds.sel(level=plevel_)
    
    return ds

def _preprocess(ds, lats, lons):
    """
    Help preprocessing files upon opening by selecting region.
    """
    # create cyclic coords for slicing across prime meridian
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.roll(longitude=int(len(ds['longitude']) / 2), roll_coords=True)
    
    # select lat/lon coords
    return ds.sel(latitude=slice(lats+4, lats-4),
                  longitude=slice(lons-4, lons+4))

def grab_forecast_hour(time, ds):
    """
    Help selecting the forecast hour for respective files
    """
    if pd.to_datetime(time).hour == 6 or pd.to_datetime(time).hour == 18:
        return int(12)
    elif pd.to_datetime(time).hour < 6:
        return pd.to_datetime(time).hour + 6
    else:
        return pd.to_datetime(time).hour - pd.Timestamp(
            ds.sel(forecast_initial_time=pd.to_datetime(time), method='ffill').forecast_initial_time.values
        ).hour

def grab_forecast_initial_time(time, ds):
    """
    Help selecting the forecast initial time for respective files
    """
    if pd.to_datetime(time).hour == 6 or pd.to_datetime(time).hour == 18:
        time = (pd.to_datetime(time) - timedelta(hours=12)).strftime('%Y-%m-%dT%H')
        return pd.Timestamp(
            ds.sel(forecast_initial_time=pd.to_datetime(time)).forecast_initial_time.values
        )
    else:
        return pd.Timestamp(
            ds.sel(forecast_initial_time=pd.to_datetime(time), method='ffill').forecast_initial_time.values
        )

def grab_time_help_era5fc(time):
    """
    Help making the forecast file string time
    """
    pt = pd.to_datetime(time)
    if pt.day == 1 and pt.hour <= 6:
        # previous month second half file
        return (pt - timedelta(days=7)).strftime('%Y-%m') + '-16T06'
    if (pt.day == 1 and pt.hour > 6) or (pt.day > 1 and pt.day < 16) or (pt.day == 16 and pt.hour <= 6):
        # first half same month file
        return pt.strftime('%Y-%m') + '-01T06'
    if (pt.day == 16 and pt.hour > 6) or (pt.day > 16):
        # second half same month file
        return pt.strftime('%Y-%m') + '-16T06'

def open_dataframe(directory_home, csv_files, file_choice):
    """
    Open and preprocess relevant csv files.
    """
    idx = isValInLst(file_choice, csv_files)[0]
    df_aew = pd.read_csv(f"{directory_home}/{csv_files[idx]}")

    # dev
    df_dev = df_aew[["developing_times","developing_lats","developing_lons"]]
    df_dev = df_dev.drop(df_dev[df_dev['developing_times']=='-999'].index)  # remove empty rows

    # non-dev
    df_non = df_aew[["nondeveloping_times","nondeveloping_lats","nondeveloping_lons"]]
    df_non = df_non.drop(df_non[df_non['nondeveloping_times']=='-999'].index)  # remove empty rows
    
    df_dev_v2 = df_dev.rename(
        columns={
            "developing_times": "times", 
            "developing_lats": "lats", 
            "developing_lons": "lons"
        },
        errors="raise"
    ).assign(label=np.ones(len(df_dev)))

    df_non_v2 = df_non.rename(
        columns={
            "nondeveloping_times": "times", 
            "nondeveloping_lats": "lats", 
            "nondeveloping_lons": "lons"
        },
        errors="raise"
    ).assign(label=np.zeros(len(df_non)))

    df_merged = pd.concat([df_dev_v2, df_non_v2]).reset_index()
    return df_merged

def tmp_file_maker(
    csv_files,
    _initpreprocess,
    plevel_,
    subfolder,
    directory_era5,
    minivar_,
    directory_home,
    file_choice
):
    """
    Create intermediary data files, writing to Zarr instead of NetCDF.
    """
    df_merged = open_dataframe(directory_home, csv_files, file_choice)
    times_filtered = df_merged["times"].unique()

    # Create a directory for your "tmp" Zarr output, if desired
    zarr_tmp_dir = f"{directory_home}{minivar_ + '_tmp'}/"
    if not os.path.exists(zarr_tmp_dir):
        os.makedirs(zarr_tmp_dir, exist_ok=True)

    for time in times_filtered:
        ym_str = pd.to_datetime(time).strftime('%Y%m')
        ymd_str = pd.to_datetime(time).strftime('%Y%m%d')

        partial_func = partial(_initpreprocess, time=time, plevel_=plevel_)

        if subfolder == "e5.oper.an.sfc":
            string_help = ym_str + '01'
        elif subfolder == "e5.oper.an.pl":
            string_help = ymd_str
        else:
            string_help = ymd_str  # Fallback if needed

        # Build file search pattern
        fpattern = f"{directory_era5}/{subfolder}/{ym_str}/e5*_{minivar_}.*.{string_help}00_*.nc"
        
        # Open the dataset
        ds = xr.open_mfdataset(fpattern, preprocess=partial_func)

        # Construct the Zarr output name
        if plevel_:
            plevel_str = f"_{int(plevel_)}"
        else:
            plevel_str = ""
        zarr_file_name = (
            f"{zarr_tmp_dir}/aew_{file_choice}{plevel_str}_"
            f"{pd.to_datetime(time).strftime('%Y%m%d%H')}.zarr"
        )

        # Save to Zarr (overwrite if it already exists)
        ds.to_zarr(zarr_file_name, mode="w")

def tmp_fcfile_maker(csv_files, subfolder, directory_era5, 
                     minivar_, directory_home, file_choice):
    """
    Create intermediary data files (forecast), writing to Zarr.
    """
    df_merged = open_dataframe(directory_home, csv_files, file_choice)
    times_filtered = df_merged["times"].unique()
    
    zarr_tmp_dir = f"{directory_home}{minivar_ + '_tmp'}/"
    if not os.path.exists(zarr_tmp_dir):
        os.makedirs(zarr_tmp_dir, exist_ok=True)

    for time in times_filtered:
        time_help = grab_time_help_era5fc(time)
        
        yrmo_help = time_help[:4] + time_help[5:7] + '/'
        file_help = (time_help[:4] + time_help[5:7] + 
                     time_help[8:10] + time_help[11:])
        
        fpattern = f"{directory_era5}{subfolder}/{yrmo_help}e5.*_{minivar_}.*.{file_help}_*.nc"
        ds = xr.open_mfdataset(fpattern)

        zarr_file_name = (
            f"{zarr_tmp_dir}/aew_{file_choice}_"
            f"{pd.to_datetime(time).strftime('%Y%m%d%H')}.zarr"
        )

        ds_sel = ds.sel(
            forecast_initial_time=grab_forecast_initial_time(time, ds),
            forecast_hour=grab_forecast_hour(time, ds)
        )

        ds_sel.to_zarr(zarr_file_name, mode="w")

def ml_file_maker(
    csv_files,
    _preprocess,
    subfolder,
    minivar_,
    directory_home,
    file_choice,
    plevel_=None
):
    """
    Create data files for ML training, writing to Zarr instead of NetCDF.
    """
    df_merged = open_dataframe(directory_home, csv_files, file_choice)

    # Directory for final ML Zarr files
    zarr_ml_dir = f"{directory_home}{minivar_}/"
    if not os.path.exists(zarr_ml_dir):
        os.makedirs(zarr_ml_dir, exist_ok=True)

    for time, lats, lons, lbl in zip(
        df_merged["times"], df_merged["lats"], df_merged["lons"], df_merged["label"]
    ):
        # partial for region subsetting
        partial_func = partial(_preprocess, lats=lats, lons=lons)

        # build file name for the intermediate tmp
        if plevel_:
            plevel_str = f"_{int(plevel_)}"
        else:
            plevel_str = ""

        tmp_zarr = (
            f"{directory_home}{minivar_ + '_tmp'}/"
            f"aew_{file_choice}{plevel_str}_"
            f"{pd.to_datetime(time).strftime('%Y%m%d%H')}.zarr"
        )

        # Open the Zarr store
        ds = xr.open_zarr(tmp_zarr, preprocess=partial_func).assign(label=lbl)

        # Construct final Zarr name
        # For each sample, you might want a separate store or a single appended store
        # Below uses a separate store for each sample, similar to the old NetCDF approach:
        final_zarr_name = (
            f"{zarr_ml_dir}/aew_{file_choice}{plevel_str}_"
            f"{pd.to_datetime(time).strftime('%Y%m%d%H')}_"
            f"{str(np.around(lats, 1))}_{str(np.around(lons, 1))}.zarr"
        )

        # Write out
        ds.to_zarr(final_zarr_name, mode="w")
