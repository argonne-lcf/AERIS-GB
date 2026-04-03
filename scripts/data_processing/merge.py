# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.

# AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions
# This work is licensed under the MIT License. See LICENSE for details.

from mpi4py import MPI

import os
from glob import glob
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import time

class AttributeSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset

    def __getattr__(self, attr):
        """Delegate attribute access to the original dataset"""
        return getattr(self.dataset, attr)

class ERA5NonResDataset(Dataset):
    def __init__(
        self,
        root: str,
        variables: list[str],
        interval: int = 1,
        split: str = "train",
        residual: bool = False,
    ):
        super().__init__()
        self.root = root
        self.files = sorted(glob(os.path.join(root, split, "*.h5")))
        self.variables = variables
        self.interval = interval
        self.residual = residual

        # TODO: fix and broadcast this to minimize overhead across nodes/gpus
        self.x_means, self.x_stds = self._setup_standardize()
        self.shape = self._load_file(
            self.files[np.random.randint(0, len(self.files))], variables
        ).shape

    @property
    def n_channels(self):
        assert len(self.shape) == 3
        return self.shape[0]

    @property
    def img_resolution(self):
        return self.shape[1], self.shape[2]

    def _load_file(self, path: str, variables: list[str]) -> np.ndarray:
        with h5py.File(path, "r") as f:
            data = {
                main_key: {
                    sub_key: np.array(value)
                    for sub_key, value in group.items()
                    if sub_key in variables + ["time"]
                }
                for main_key, group in f.items()
                if main_key in ["input"]
            }
            return np.stack([data["input"][v] for v in variables], axis=0)

    def _load_and_stack(self, filename: str) -> np.ndarray:
        with np.load(os.path.join(self.root, filename)) as data:
            return np.stack([data[v] for v in self.variables], axis=0).reshape(-1, 1, 1)

    def _setup_standardize(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_means = self._load_and_stack("normalize_mean.npz")
        x_stds = self._load_and_stack("normalize_std.npz")


        return x_means, x_stds

    def standardize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_means) / self.x_stds

    def unstandardize_x(self, x: np.ndarray) -> np.ndarray:
        return x * self.x_stds + self.x_means
    
    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.load(os.path.join(self.root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(self.root, "lon.npy")).astype(np.float32)
        return lat, lon

    def get_time(self, idx: int) -> np.datetime64:
        with h5py.File(self.files[idx], "r") as f:
            timestamp = f["input"]["time"][()]
            return np.datetime64(timestamp.decode("utf-8"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._load_file(self.files[idx], self.variables)
        return x



if __name__ == "__main__":
    print("begin", flush=True)
    rank = int(MPI.COMM_WORLD.Get_rank())
    world_size = int(MPI.COMM_WORLD.Get_size())

    root = "/flare/datasets/wb2/0.25deg_1_step_6hr_h5df_fix_bug"
    split = "test"
    samples = 1464#58440, 1464
    fname = "/flare/SAFS/vhat/data/enchanced2_merged_test.hdf5"
    #variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200", "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500", "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "geopotential_1000", "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "u_component_of_wind_1000", "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "v_component_of_wind_1000", "temperature_50", "temperature_100", "temperature_150", "temperature_200", "temperature_250", "temperature_300", "temperature_400", "temperature_500", "temperature_600", "temperature_700", "temperature_850", "temperature_925", "temperature_1000", "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200", "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500", "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925", "specific_humidity_1000", "toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask"]
    variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200", "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500", "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "geopotential_1000", "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "u_component_of_wind_1000", "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "v_component_of_wind_1000", "temperature_50", "temperature_100", "temperature_150", "temperature_200", "temperature_250", "temperature_300", "temperature_400", "temperature_500", "temperature_600", "temperature_700", "temperature_850", "temperature_925", "temperature_1000", "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200", "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500", "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925", "specific_humidity_1000", "sea_surface_temperature" ,"toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask"]

    ds = ERA5NonResDataset(root, variables, 1, split, residual=False)
    assert len(ds) == samples, (len(ds), samples)
    MPI.COMM_WORLD.barrier()

    indices = range(samples)[rank::world_size]
    dataset = AttributeSubset(ds, indices=indices)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=7,
        prefetch_factor=3,
        persistent_workers=True,
    )
    i_dataloader = iter(dataloader)

    print("begin init", flush=True)
    rewrite = True
    if rank == 0 and rewrite:
        write_f = h5py.File(fname, 'w')
        dset_write = write_f.create_dataset("input", (samples,73,721,1440), 'f')
        write_f.close()
    time.sleep(3)
    MPI.COMM_WORLD.barrier()
    print("done init", flush=True)

    wf = h5py.File(fname, 'a')
    dset_write = wf["input"]
    time.sleep(3)
    MPI.COMM_WORLD.barrier()
    items = 0
    for i in indices:
        start0 = time.time()
        #f"/flare/datasets/wb2/testing/individual_residuals/{i}.hdf5"
        #x = ds.__getitem__(i)
        x = next(i_dataloader)[0]
        start1 = time.time()
        #print("read", dset_r.shape,flush=True)
        dset_write[i,:,:,:] = x[:,:,:]
        end = time.time()
        print(f"{i+1}/{samples}", "processed in {:.2f} {:.2f}".format(start1-start0, end-start1), flush=True)
        if items == 100:
            wf.close()
            wf = h5py.File(fname, 'a')
            dset_write = wf["input"]
            items = 0
        items += 1
    wf.close()
    print("done")

