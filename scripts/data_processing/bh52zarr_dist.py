# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.

# AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions
# This work is licensed under the MIT License. See LICENSE for details.

import argparse
import logging
import os
import socket
import time
from urllib.parse import urlparse

import dask
import h5py
import numpy as np
import xarray as xr
import zarr
from dask.distributed import Client, progress
from einops import rearrange

VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_50",
    "geopotential_100",
    "geopotential_150",
    "geopotential_200",
    "geopotential_250",
    "geopotential_300",
    "geopotential_400",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "geopotential_1000",
    "u_component_of_wind_50",
    "u_component_of_wind_100",
    "u_component_of_wind_150",
    "u_component_of_wind_200",
    "u_component_of_wind_250",
    "u_component_of_wind_300",
    "u_component_of_wind_400",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "u_component_of_wind_1000",
    "v_component_of_wind_50",
    "v_component_of_wind_100",
    "v_component_of_wind_150",
    "v_component_of_wind_200",
    "v_component_of_wind_250",
    "v_component_of_wind_300",
    "v_component_of_wind_400",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "v_component_of_wind_1000",
    "temperature_50",
    "temperature_100",
    "temperature_150",
    "temperature_200",
    "temperature_250",
    "temperature_300",
    "temperature_400",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "temperature_1000",
    "specific_humidity_50",
    "specific_humidity_100",
    "specific_humidity_150",
    "specific_humidity_200",
    "specific_humidity_250",
    "specific_humidity_300",
    "specific_humidity_400",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
    "specific_humidity_1000",
]

from aeris.utils.io import compress_variables

parser = argparse.ArgumentParser(description="Convert HDF5 to Zarr.")
parser.add_argument(
    "-i", "--input", type=str, required=True, help="Path to input HDF5 file"
)
parser.add_argument(
    "-o", "--output", type=str, required=True, help="Output directory for Zarr store"
)
parser.add_argument(
    "-n", "--name", type=str, required=True, help="Name of the Zarr dataset"
)
parser.add_argument(
    "--dask-scheduler", type=str, required=True, help="Path to Dask scheduler file"
)
parser.add_argument(
    "--interval", type=int, default=12, help="Interval between initials (default: 12)"
)
parser.add_argument(
    "--workers", type=int, default=16, help="Number of worker processes (default: 16)"
)
parser.add_argument(
    "--batch-size", type=int, default=64, help="Number of steps per batch (default: 4)"
)

parser.add_argument(
    "--ws_y", type=int, default=30, help="Number of steps per batch (default: 4)"
)
parser.add_argument(
    "--ws_x", type=int, default=30, help="Number of steps per batch (default: 4)"
)


def load_batch(
    initial_idx: int, batch_start: int, batch_end: int, input_path: str, ws_y=30, ws_x=30
) -> np.ndarray:
    for i in range(10):
        with h5py.File(input_path, "r") as f:
            data = f["input"][:, :, initial_idx, batch_start:batch_end, :, :, :]
            try:
                data = rearrange(
                    data,
                    "d e s c (wc_y ws_y) (wc_x ws_x) -> s c (wc_y d ws_y) (wc_x e ws_x)",
                    ws_y=ws_y,
                    ws_x=ws_x,
                )
                data = np.pad(data, pad_width=((0, 0), (0, 0), (1, 0), (0, 0)), mode="reflect")
                return data  # [bs, n_channels, n_lat, n_lon]
            except:
                dask.distributed.print("retrying:", initial_idx, batch_start, batch_end, input_path)
                continue
    raise NotImplementedError



def process_batch(
    initial_idx: int, batch_start: int, batch_end: int, input_path: str, zarr_store: str, ws_y=30, ws_x=30
):
    data = load_batch(initial_idx, batch_start, batch_end, input_path, ws_y, ws_x)
    if type(data)==type(None):
        raise NotImplementedError
    zarr_group = zarr.open_group(zarr_store, mode="a")

    for var, levels in compress_variables(VARS).items():
        if not levels:
            channel_idx = VARS.index(var)
            var_data = data[:, channel_idx, :, :]  # [bs, n_lat, n_lon]
            zarr_group[var][initial_idx, batch_start:batch_end, :, :] = var_data
        else:
            sorted_levels = sorted(levels)
            channel_indices = [VARS.index(f"{var}_{lev}") for lev in sorted_levels]
            var_data = data[:, channel_indices, :, :]  # [bs, n_levels, n_lat, n_lon]
            zarr_group[var][initial_idx, batch_start:batch_end, :, :, :] = var_data


def main(args):
    print("Starting conversion...")
    with h5py.File(args.input, "r") as f:
        shapes = {}

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                shapes[name] = obj.shape

        f.visititems(visitor)
        _, _, n_initials, n_steps, n_channels, _, _ = shapes["input"]

    if n_channels != len(VARS):
        raise ValueError(f"Expected {len(VARS)} channels, got {n_channels}")

    start_time = np.datetime64("2020-01-01T00:00:00.000000000")
    time_coord = start_time + np.arange(n_initials) * np.timedelta64(args.interval, "h")
    pred_td = (np.arange(n_steps) * np.timedelta64(6, "h")).astype("timedelta64[ns]")

    compressed_variables = compress_variables(VARS)
    compressed_variables.pop("time", None)

    root = "/flare/datasets/wb2/0.25deg_1_step_6hr_h5df_fix_bug/"
    lat = np.load(os.path.join(root, "lat.npy")).astype(np.float32)
    lon = np.load(os.path.join(root, "lon.npy")).astype(np.float32)
    n_lat, n_lon = len(lat), len(lon)

    n_levels = max((len(levels) for levels in compressed_variables.values()), default=0)

    coords = {
        "time": (("time",), time_coord),
        "prediction_timedelta": (("prediction_timedelta",), pred_td),
        "latitude": (("latitude",), lat),
        "longitude": (("longitude",), lon),
        "level": (("level",), np.arange(n_levels, dtype=np.int32)),
    }

    ofile = os.path.join(args.output, f"{args.name}.zarr")
    os.makedirs(args.output, exist_ok=True)

    coords_ds = xr.Dataset(coords=coords)
    coords_ds.to_zarr(ofile, mode="w")

    zarr_group = zarr.open_group(ofile, mode="a")
    for var, levels in compressed_variables.items():
        has_levels = bool(levels)
        shape = (
            (n_initials, n_steps, n_lat, n_lon)
            if not has_levels
            else (n_initials, n_steps, len(levels), n_lat, n_lon)
        )
        chunks = (
            (1, 1, n_lat, n_lon)
            if not has_levels
            else (1, 1, len(levels), n_lat, n_lon)
        )
        ds = zarr_group.create_dataset(var, shape=shape, chunks=chunks, dtype="f4")
        ds.attrs["_ARRAY_DIMENSIONS"] = (
            ["time", "prediction_timedelta", "latitude", "longitude"]
            if not has_levels
            else ["time", "prediction_timedelta", "level", "latitude", "longitude"]
        )

    tasks = []
    for i in range(n_initials):
        for batch_start in range(0, n_steps, args.batch_size):
            batch_end = min(batch_start + args.batch_size, n_steps)
            tasks.append(
                dask.delayed(process_batch)(
                    i, batch_start, batch_end, args.input, ofile, args.ws_y, args.ws_x
                )
            )
    print(f"Processing {len(tasks)} tasks...")

    with Client(scheduler_file=args.dask_scheduler) as client:
        scheduler_info = client.scheduler_info()
        nworkers = len(scheduler_info["workers"])

        # Extract IP from dashboard link and resolve hostname
        parsed_url = urlparse(client.dashboard_link)
        ip = parsed_url.hostname
        try:
            hostname = socket.gethostbyaddr(ip)[0]
        except socket.herror:
            hostname = ip  # fallback to IP if reverse DNS fails

        dashboard_url = client.dashboard_link.replace(ip, hostname)
        print(f"Initialized Client: {nworkers} workers, link {dashboard_url}")

        futures = client.compute(tasks)
        progress(futures)
        client.gather(futures)

    zarr.consolidate_metadata(ofile)
    print(f"Zarr store created at {ofile}")


if __name__ == "__main__":
    start = time.time()
    args = parser.parse_args()
    main(args)
    print(f"Elapsed time: {time.time() - start:.2f} seconds")
