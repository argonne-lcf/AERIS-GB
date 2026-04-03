# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.

# AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions
# This work is licensed under the MIT License. See LICENSE for details.

import os
from glob import glob
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import time
import gc
from einops import rearrange

from torch.utils.data import Subset
# ----------------------------------------------------------------------------
# Subset for torch.utils.data.Dataset that allows for a subset of the dataset
# with support for attribute delegation to the original dataset, e.g., len().
class AttributeSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset

    def __getattr__(self, attr):
        """Delegate attribute access to the original dataset"""
        return getattr(self.dataset, attr)

class ERA5HDF5InferenceDataset(Dataset):
    #Non-redisual monolithic (preprocessed single file) dataset
    def __init__(
        self,
        variables,
        root = '/flare/datasets/wb2/0.25deg_1_step_6hr_h5df_fix_bug/',
        path = '/flare/datasets/wb2/testing/monolithic_striped/monolithic2.hdf5',
        interval = 6,
        rollout_count = 1,
    ):
        super().__init__()
        self.variables = variables
        self.root = root
        self.path = path
        self.interval = interval
        self.rollout_count = rollout_count
        self.samples = (1464-(interval//6))
        assert interval==24 or not "sea_surface_temperature" in variables, "Sea surface temperature only implemented for 24h prediction"
        self.channels = len(variables)

        self.shape = [self.channels,721,1440]
        self.read = 0
        self.sst_mask = None
        self._setup_standardize()

        self.file = h5py.File(path, 'r')
        self.ds = self.file["input"]

    @property
    def n_channels(self):
        assert len(self.shape) == 3
        return self.shape[0]

    @property
    def img_resolution(self):
        return self.shape[1], self.shape[2]

    @property
    def variables(self):
        #variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200", "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500", "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "geopotential_1000", "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "u_component_of_wind_1000", "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "v_component_of_wind_1000", "temperature_50", "temperature_100", "temperature_150", "temperature_200", "temperature_250", "temperature_300", "temperature_400", "temperature_500", "temperature_600", "temperature_700", "temperature_850", "temperature_925", "temperature_1000", "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200", "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500", "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925", "specific_humidity_1000", "sea_surface_temperature" ,"toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask"]
        return self._variables
    @variables.setter
    def variables(self,value):
        self._variables = value

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.load(os.path.join(self.root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(self.root, "lon.npy")).astype(np.float32)
        return lat, lon
    
    def _load_and_stack(self, filename: str) -> np.ndarray:
        root = self.root
        with np.load(os.path.join(root, filename)) as data:
            return np.stack([data[v] for v in self.variables], axis=0)
    
    def _setup_standardize(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.x_means = self._load_and_stack("normalize_mean.npz")
        self.x_stds = self._load_and_stack("normalize_std.npz")

        self.t_stds = self._load_and_stack(f"normalize_diff_std_{self.interval}.npz")
        self.t_means = self._load_and_stack(f"normalize_diff_mean_{self.interval}.npz")

    def standardize_x(self, x: np.ndarray, start_ch=0, end_ch=None, channel_dim=0) -> np.ndarray:
        shape = [1,1,1]
        shape[channel_dim] = -1
        out = (x - self.x_means[start_ch:end_ch].reshape(shape)) / self.x_stds[start_ch:end_ch].reshape(shape)
        return out

    def unstandardize_x(self, x: np.ndarray, start_ch=0, end_ch=None, channel_dim=0) -> np.ndarray:
        shape = [1,1,1]
        shape[channel_dim] = -1
        return x * self.x_stds[start_ch:end_ch].reshape(shape) + self.x_means[start_ch:end_ch].reshape(shape)

    def standardize_t(self, t: np.ndarray, start_ch=0, end_ch=None, channel_dim=0) -> np.ndarray:
        shape = [1,1,1]
        shape[channel_dim] = -1
        return (t - self.t_means[start_ch:end_ch].reshape(shape)) / self.t_stds[start_ch:end_ch].reshape(shape)

    def unstandardize_t(self, t: np.ndarray, start_ch=0, end_ch=None, channel_dim=0) -> np.ndarray:
        shape = [1,1,1]
        shape[channel_dim] = -1
        return t * self.t_stds[start_ch:end_ch].reshape(shape) + self.t_means[start_ch:end_ch].reshape(shape)
    
    def get_time(self, idx: int) -> np.datetime64:
        with h5py.File(self.files[idx], "r") as f:
            timestamp = f["input"]["time"][()]
        return np.datetime64(timestamp.decode("utf-8"))

    def __len__(self) -> int:
        return self.samples  # last files dont have a target
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.read > 10:
            gc.collect()
            self.read = 0

        x_ind = idx
        dt = (self.interval//6)
        
        x = np.array(self.ds[x_ind,:,:,:])
        self.read += 1

        #Labels needed for local eval
        if "sea_surface_temperature" in self.variables:

            sst_ind = self.variables.index("sea_surface_temperature")
            #np.copyto(x[sst_ind], np.nanmin(x[sst_ind]), where=np.isnan(x)[sst_ind])
            sst = x[sst_ind]
            sst = rearrange(sst[1:].copy(), "(wc_y d ws_y) (wc_x e ws_x) -> d e (wc_y ws_y) (wc_x ws_x)", d=4, e=4, ws_y=60, ws_x=60)
            for i in range(4):
                for j in range(4):
                    np.copyto(sst[i,j], np.nanmin(sst[i,j]), where=np.isnan(sst[i,j]))
            sst = rearrange(sst, "d e (wc_y ws_y) (wc_x ws_x) -> (wc_y d ws_y) (wc_x e ws_x)", ws_y=60, ws_x=60)
            np.copyto(x[sst_ind][1:], sst)

            
        
        x = torch.from_numpy(self.standardize_x(x)).float()  
        labels = np.array(self.ds[x_ind:x_ind+(self.rollout_count+1)*dt:dt,:,:,:])

        if "sea_surface_temperature" in self.variables:
            np.copyto(labels[:,sst_ind], np.nanmin(labels[:,sst_ind]), where=np.isnan(labels[:,sst_ind]))
        
        return x, labels
    

class AERIS_SP_ERA5_Data():
    def __init__(
            self,
            inner_dataset,
            dataloader,
            out_channels,
            rollout_steps,
            device_mesh,
            sp_group_gloo,
            wp_dims,
            image_shape,
            window_size,
            device
        ):
        self.inner_dataset = inner_dataset
        self.out_channels = out_channels
        self.rollout_steps = rollout_steps
        self.mesh = device_mesh
        self.sp_rank = self.mesh.get_local_rank(mesh_dim=1)
        self.sp_group = self.mesh.get_group(mesh_dim=1)
        self.SP = torch.distributed.get_world_size(group=self.sp_group)
        self.sp_group_gloo = sp_group_gloo
        self.device=device
        self.wp_dims = wp_dims
        self.wp = wp_dims[0] > 1 or wp_dims[1] > 1
        self.image_shape = image_shape
        self.window_size = window_size

        if dataloader is not None:
            self.dataloader = iter(dataloader)
        self.lsm=None
        self.gp=None
    
    def __len__(self) -> int:
        return self.inner_dataset.samples  # last files dont have a target
    
    def get_gp(self):
        if self.gp is None:
            gp_ind = self.inner_dataset.variables.index("geopotential_at_surface")
            gp = self.inner_dataset[0][0].unsqueeze(0)[:,gp_ind:gp_ind+1, 1:].numpy()#Cut out north pole
            gp = torch.from_numpy(gp).to(device="cpu", dtype=torch.float).float()
            #gp = torch.from_numpy(self.inner_dataset.standardize_x(gp, gp_ind, gp_ind+1)).to(device="cpu", dtype=torch.float).float()
            #print("gp", self.inner_dataset.x_stds[gp_ind], self.inner_dataset.x_means[gp_ind], gp, flush=True)
            self.gp = self.shard_tensor(gp,(720*1440),1)
            #print("gp", gp)
        return self.gp
    
    def get_lsm(self):
        if self.lsm is None:
            lsm_ind = self.inner_dataset.variables.index("land_sea_mask")
            lsm = self.inner_dataset[0][0].unsqueeze(0)[:,lsm_ind:lsm_ind+1, 1:].numpy()#Cut out north pole
            lsm = torch.from_numpy(lsm).to(device="cpu", dtype=torch.float).float()
            #lsm = torch.from_numpy(self.inner_dataset.standardize_x(lsm, lsm_ind, lsm_ind+1)).to(device="cpu", dtype=torch.float).float()
            self.lsm = self.shard_tensor(lsm,(720*1440),1)
        return self.lsm
    
    def standardize_x(self,*args,**kwargs):
        return self.inner_dataset.standardize_x(*args,**kwargs)

    def standardize_t(self,*args,**kwargs):
        return self.inner_dataset.standardize_t(*args,**kwargs)

    def unstandardize_x(self,*args,**kwargs):
        return self.inner_dataset.unstandardize_x(*args,**kwargs)

    def unstandardize_t(self,*args,**kwargs):
        return self.inner_dataset.unstandardize_t(*args,**kwargs)

    def gather_tensor(self, tensor):
        if self.sp_rank == 0:
            input_list = [torch.zeros_like(tensor) for _ in range(self.SP)]
        else:
            input_list = None
        
        dst = torch.distributed.get_global_rank(self.sp_group, group_rank=0)
        torch.distributed.gather(tensor.contiguous(), gather_list=input_list, dst=dst, group=self.sp_group_gloo)
        time.sleep(0.5)

        if self.sp_rank == 0:
            if self.wp:
                out = torch.stack(input_list,dim=1)
                wp_y, wcl_y, ws_y, wp_x, wcl_x, ws_x = self.wp_dims[0], self.image_shape[0]//self.window_size[0]//self.wp_dims[0], self.window_size[0], self.wp_dims[1], self.image_shape[1]//self.window_size[1]//self.wp_dims[1], self.window_size[1]
                out = rearrange(out, "b (wp_y wp_x) (wcl_y ws_y wcl_x ws_x) c -> b (wp_y wcl_y ws_y wp_x wcl_x ws_x) c", wp_y=wp_y, wcl_y=wcl_y, ws_y=ws_y, wp_x=wp_x, ws_x=ws_x) 
            else:
                raise NotImplementedError
                out = torch.cat(input_list,dim=1)
        else:
            out = None
        return out 

    def shard_tensor(self, tensor: torch.Tensor, dim_size: int, channels, batch=1, dtype=torch.float32):
        assert dim_size%self.SP == 0, f"can not shard tensor dimension {dim_size} to {self.SP}"
        sharded = torch.zeros((batch,dim_size//self.SP,channels), dtype=dtype, device="cpu")
        if self.sp_rank == 0:
            if self.wp:
                """
                ranks:
                0,  1,  2,  3
                4,  5,  6,  7
                8,  9,  10, 11
                12, 13, 14, 15
                """
                wp_y, ws_y, wp_x, ws_x = self.wp_dims[0], self.window_size[0], self.wp_dims[1], self.window_size[1]
                #tensor = rearrange(tensor, "b c (wp_y wcl_y ws_y) (wp_x wcl_x ws_x) -> b (wp_y wp_x) (wcl_y ws_y wcl_x ws_x) c", wp_y=wp_y, ws_y=ws_y, wp_x=wp_x, ws_x=ws_x)
                tensor = rearrange(tensor, "b c (wp_y wcl_y ws_y) (wp_x wcl_x ws_x) -> b (wp_y wp_x) (wcl_y ws_y wcl_x ws_x) c", wp_y=wp_y, ws_y=ws_y, wp_x=wp_x, ws_x=ws_x)
                tensor_list = [item[:,0,:,:].contiguous() for item in tensor.chunk(wp_y*wp_x, dim=1)]
            else:
                raise NotImplementedError
                tensor = rearrange(tensor, "b c y x -> b (y x) c") #batch, height, width, channel -> batch sequence channel -> partition
                tensor_list = [item.contiguous() for item in tensor.chunk(self.SP, dim=1)]
        else:
            tensor_list = None
        src = torch.distributed.get_global_rank(self.sp_group, group_rank=0)
        #print(tensor)
        torch.distributed.scatter(sharded, scatter_list=tensor_list, src=src, group=self.sp_group_gloo)
        time.sleep(0.3)
        torch.xpu.synchronize()
        #sharded = DTensor.from_local(sharded.to(self.device), self.mesh, placements=[Shard(0),Shard(1)])
        return sharded

    
    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sp_rank==0:
            X, labels = next(self.dataloader)#B,C,H,W
            X = X[:,:self.out_channels,1:] #Cut out north pole, solar rad, geopotential_at_surface and land_sea_mask

            rad_ind = self.inner_dataset.variables.index("toa_incident_solar_radiation")

            rad_T = labels[0,:,rad_ind:rad_ind+1,1:].numpy()
            rad_T = torch.from_numpy(self.inner_dataset.standardize_x(rad_T, rad_ind, rad_ind+1)).to(device="cpu", dtype=torch.float).float()#standardized
            X_sharded = self.shard_tensor(X,(720*1440),self.out_channels, batch=1)
            rad_T = self.shard_tensor(rad_T,(720*1440), 1, batch=self.rollout_steps+1)
            return X_sharded, rad_T, X
        else:
            X = self.shard_tensor(None,(720*1440),self.out_channels, batch=1)
            rad_T = self.shard_tensor(None,(720*1440), 1, batch=self.rollout_steps+1)
            return X, rad_T, None



        




