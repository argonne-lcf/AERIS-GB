# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.

# AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions
# This work is licensed under the MIT License. See LICENSE for details.

"""
Example usage:

python -m aeris.plotting.rollout \
    --prediction_path=/home/jstock/md/aeris/results/era5-unet-xl/3635595.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/output/output-32i-56s-8m.zarr \
    --variable=t2m \
    --name=unet-xl \
    --time_start=2020-01-01
"""

import argparse
import os
import gc

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import h5py
import xarray as xr
from matplotlib.animation import FuncAnimation
from einops import rearrange

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prediction_path", type=str, required=True, help="Prediction zarr file"
)
parser.add_argument("--target_path", type=str, required=False, help="Target zarr file")
parser.add_argument("--diffusion", action="store_true", help="Plot output from diffusion rollout")
parser.add_argument("--eval-training-data", action="store_true", help="Plot training data")
parser.add_argument("--training-data", action="store_true", help="Plot training data")
parser.add_argument("--wp-training-data", action="store_true", help="Plot training data")
parser.add_argument("--full-format", action="store_true", help="Plot the difference")
parser.add_argument("--ensemble-mean", action="store_true", help="Plot the difference")
parser.add_argument("--difference", action="store_true", help="Plot the difference")
parser.add_argument("--initial", type=int, default=0, help="Initial condition index")
parser.add_argument("--ws_y", type=int, default=30, help="Initial condition index")
parser.add_argument("--ws_x", type=int, default=30, help="Initial condition index")

# name
parser.add_argument("--name", type=str, default="unet", help="Model name")

VARIABLES = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "q700": "specific_humidity_700",
    "z500": "geopotential_500",
    "toa_incident_solar_radiation": "toa_incident_solar_radiation",
    "geopotential_at_surface": "geopotential_at_surface",
    "land_sea_mask": "land_sea_mask",
    "sst": "sea_surface_temperature",
}

parser.add_argument(
    "--variable",
    type=str,
    choices=list(VARIABLES.keys()),
    default="t2m",
    help="Variable to plot",
)
parser.add_argument("--member", type=int, default=-1, help="Ensemble member to plot")
parser.add_argument(
    "--time_start", type=str, default="2020-01-01", help="Start time for evaluation"
)


def animate(args, data, lats, lons, variable, name, time_start):
    fig = plt.figure(figsize=(4, 2))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    if variable=="sst":
        vmin=273
        vmax=305
    elif not args.training_data and variable=="t2m":
        vmin=240
        vmax=320
    elif not args.training_data and variable=="msl":
        vmin=98000
        vmax=103000
    elif not args.training_data and variable=="q700":
        vmin=-0.001
        vmax=0.012
    elif not args.training_data and variable=="z500":
        vmin=48000
        vmax=58000
    else:
        vmin=np.nanmin(data)
        vmax=np.nanmax(data)

    im = ax.pcolormesh(
        lons,
        lats,
        data[0],
        transform=ccrs.PlateCarree(),
        cmap="gist_ncar",
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    )
    #norm=mcolors.Normalize(vmin=data.min(), vmax=data.max()),
    #norm=mcolors.Normalize(vmin=-0.001, vmax=0.012),
    ax.coastlines()
    cbar = plt.colorbar(im, ax=ax, shrink=1, pad=0.02)

    step_text = ax.text(
        70, 85, "Step: 0", ha="left", va="top", weight="bold", fontsize=11, color="w"
    )
    ax.set_title(f"{name}, {variable}", loc="left", fontsize=11)

    generator = lambda data, steps: ((i, frame) for i, frame in enumerate(data[:steps]))

    def update(frame_data):
        i, frame = frame_data
        step_text.set_text(f"Step: {i}")
        im.set_array(frame.ravel())
        return [im, step_text]

    fps = 10
    steps = len(data)
    save_as = f"media/{name}-{variable}.gif"

    ani = FuncAnimation(
        fig,
        update,
        frames=generator(data, steps),
        interval=1000 / fps,
        blit=True,
        cache_frame_data=False,
    )

    fig.tight_layout()
    ani.save(save_as, fps=fps, dpi=300)

def eval_train_data(path, args, coords=False):
    vars = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200", "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500", "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "geopotential_1000", "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "u_component_of_wind_1000", "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "v_component_of_wind_1000", "temperature_50", "temperature_100", "temperature_150", "temperature_200", "temperature_250", "temperature_300", "temperature_400", "temperature_500", "temperature_600", "temperature_700", "temperature_850", "temperature_925", "temperature_1000", "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200", "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500", "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925", "specific_humidity_1000", "sea_surface_temperature", "toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask"]
    ind = vars.index("sea_surface_temperature")
    data = np.array(h5py.File(path)["input"][:,ind,:,:])#(samples,73,721,1440)

    residual = data[4:] - data[:-4]
    
    np.copyto(data, np.nanmean(data), where=np.isnan(data))
    print("data.mean()", data.mean())
    #np.copyto(data, np.nanstd(data), where=np.isnan(data))
    #print("data.mean()", data.std())
    np.copyto(residual, np.nanmean(residual), where=np.isnan(residual))
    print("residual.mean()", residual.mean())
    #np.copyto(residual, np.nanstd(residual), where=np.isnan(residual))
    #print("residual.std()", residual.std())
    return data

def load_train_data(path, args, coords=False):
    vars = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200", "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500", "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "geopotential_1000", "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "u_component_of_wind_1000", "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "v_component_of_wind_1000", "temperature_50", "temperature_100", "temperature_150", "temperature_200", "temperature_250", "temperature_300", "temperature_400", "temperature_500", "temperature_600", "temperature_700", "temperature_850", "temperature_925", "temperature_1000", "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200", "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500", "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925", "specific_humidity_1000", "sea_surface_temperature", "toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask"]
    ind = vars.index(VARIABLES[args.variable])
    data = h5py.File(path)["input"][:180*4:4,ind,:,:]#(samples,73,721,1440)
    print("data0", data[0])
    print("data1", data[1])
    if coords:
        root = "/flare/SAFS/data/0.25deg_1_step_6hr_h5df_fix_bug/"
        lat = np.load(os.path.join(root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(root, "lon.npy")).astype(np.float32)
        return data, lat, lon
    return data

def load_wp_train_data(path, args, coords=False):
    vars = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200", "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500", "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "geopotential_1000", "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "u_component_of_wind_1000", "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "v_component_of_wind_1000", "temperature_50", "temperature_100", "temperature_150", "temperature_200", "temperature_250", "temperature_300", "temperature_400", "temperature_500", "temperature_600", "temperature_700", "temperature_850", "temperature_925", "temperature_1000", "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200", "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500", "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925", "specific_humidity_1000", "sea_surface_temperature" ,"toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask"]
    ind = vars.index(VARIABLES[args.variable])
    data = []
    for wp_y in range(2):
        out_dat = []
        for wp_x in range(2):
            out_dat.append(h5py.File(path+f"/y_{wp_y}_x_{wp_x}.hdf5", 'r')["input"][:50,ind,:,:])
        data.append(np.stack(out_dat))
    data = np.stack(data)
    data = rearrange(data, "d e s (wc_y ws_y) (wc_x ws_x) -> s (wc_y d ws_y) (wc_x e ws_x)", ws_y=args.ws_y, ws_x=args.ws_x)
    print("data0", data[0])
    print("data1", data[1])
    if coords:
        root = "/flare/SAFS/data/0.25deg_1_step_6hr_h5df_fix_bug/"
        lat = np.load(os.path.join(root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(root, "lon.npy")).astype(np.float32)
        return data, lat, lon
    return data

def load_data(file, args, coords=False):
    # assert file exists
    assert os.path.exists(file), f"File {file} does not exist"

    #ds = xr.open_zarr(file, decode_timedelta=False)
    #data = ds[VARIABLES[args.variable]].sel(
    #    time=np.array(args.time_start, dtype="datetime64[ns]"), number=args.member
    #)
    vars = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200", "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500", "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "geopotential_1000", "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "u_component_of_wind_1000", "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "v_component_of_wind_1000", "temperature_50", "temperature_100", "temperature_150", "temperature_200", "temperature_250", "temperature_300", "temperature_400", "temperature_500", "temperature_600", "temperature_700", "temperature_850", "temperature_925", "temperature_1000", "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200", "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500", "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925", "specific_humidity_1000", "sea_surface_temperature" ,"toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask"]
    ind = vars.index(VARIABLES[args.variable])

    #if args.variable == "z500":
    #    data = data.sel(level=7)

    if True:
        read_f = h5py.File(file, 'r')

        #dset_write = write_f.create_dataset("input", (tot_samples, members,steps+1,69,721,1440), 'f')
        #print(write_f["input"])
        #np.ones((tot_samples, steps + 1, 69,720//engine.WP_Y,1440//engine.WP_X), dtype=np.float32)
        #engine.WP_Y,engine.WP_X,len(stride_indices),members
        if args.ensemble_mean:
            data = read_f["input"][:,:,args.initial,:,:,ind,:,:].mean(axis=3)#(engine.WP_Y,engine.WP_X,stride_ic,rollout_steps+1,members,out_channels,720//engine.WP_Y,1440//engine.WP_X)
        elif args.full_format:
            data = read_f["input"][:,:,args.initial,:,args.member,ind,:,:]#(engine.WP_Y,engine.WP_X,stride_ic,rollout_steps+1,members,out_channels,720//engine.WP_Y,1440//engine.WP_X)
        elif args.diffusion:
            data = read_f["input"][:,:,args.initial,args.member,:,ind,:,:]#engine.WP_Y,engine.WP_X,len(stride_indices),members,steps,69,720//engine.WP_Y,1440//engine.WP_X
        elif args.member > -1:
            data = read_f["input"][:,:,args.initial,args.member,:,ind,:,:]#engine.WP_Y,engine.WP_X,tot_samples,steps+1,69,720//engine.WP_Y,1440//engine.WP_X
        else:
            data = read_f["input"][:,:,args.initial,:,ind,:,:]#engine.WP_Y,engine.WP_X,tot_samples,steps+1,69,720//engine.WP_Y,1440//engine.WP_X


        #(4, 4, 51, 180, 360)
        data = rearrange(data, "d e s (wc_y ws_y) (wc_x ws_x) -> s (wc_y d ws_y) (wc_x e ws_x)", ws_y=args.ws_y, ws_x=args.ws_x)
        print("data0", data[0])
        print("data1", data[1])

    if coords:
        root = "/flare/SAFS/data/0.25deg_1_step_6hr_h5df_fix_bug/"
        lat = np.load(os.path.join(root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(root, "lon.npy")).astype(np.float32)
        return data, lat, lon
    return data


def main(args):
    if args.eval_training_data:
        eval_train_data(args.prediction_path, args, coords=True)
    else:
        if args.training_data:
            pred, lats, lons = load_train_data(args.prediction_path, args, coords=True)
        elif args.wp_training_data:
            pred, lats, lons = load_wp_train_data(args.prediction_path, args, coords=True)
        else:
            pred, lats, lons = load_data(args.prediction_path, args, coords=True)
            if args.target_path:
                target = load_data(args.target_path, args)

        animate(args, pred, lats, lons, args.variable, args.name, args.time_start)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
