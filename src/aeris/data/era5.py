import os
from glob import glob
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import time
import gc


class ERA5Dataset(Dataset):
    #Legacy implementation for file per sample approach
    def __init__(
        self,
        root: str,
        variables: list[str],
        interval: int = 1,
        split: str = "train",
    ):
        super().__init__()
        self.root = root
        self.files = sorted(glob(os.path.join(root, split, "*.h5")))
        self.variables = variables
        self.interval = interval

        means = np.load(os.path.join(root, f"normalize_mean.npz"))
        self.means = np.stack([means[v] for v in variables], axis=0).reshape(-1, 1, 1)
        stds = np.load(os.path.join(root, f"normalize_std.npz"))
        self.stds = np.stack([stds[v] for v in variables], axis=0).reshape(-1, 1, 1)

        self.ind_mask = np.array([0])
        # TODO: fix and broadcast this to minimize overhead across nodes/gpus
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

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.load(os.path.join(self.root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(self.root, "lon.npy")).astype(np.float32)
        return lat, lon

    def get_time(self, idx: int) -> np.datetime64:
        with h5py.File(self.files[idx], "r") as f:
            timestamp = f["input"]["time"][()]
        return np.datetime64(timestamp.decode("utf-8"))
    
    def set_ind_mask(self,ind_mask, loaded_shape):
        #print("setting ind_mask", ind_mask.sum(), ind_mask.size, ind_mask.shape, ind_mask)
        self.ind_mask = ind_mask
        self.loaded_shape = loaded_shape
    #[:,0:,:,:][self.ind_slice]
    def _load_file(self, path: str, variables: list[str]) -> np.ndarray:
        with h5py.File(path, "r") as f:
            
            if self.ind_mask.size > 1:
                #print("loading with ind_mask")
                #mask_indices = np.where(self.ind_mask)
                data = {
                    main_key: {
                        sub_key: np.array(value[()][self.ind_mask]).reshape(self.loaded_shape)
                        for sub_key, value in group.items()
                        if sub_key in variables
                    }
                    for main_key, group in f.items()
                    if main_key in ["input"]
                }
            else:
                #print("loading without ind_mask")
                data = {
                    main_key: {
                        sub_key: np.array(value)
                        for sub_key, value in group.items()
                        if sub_key in variables + ["time"]
                    }
                    for main_key, group in f.items()
                    if main_key in ["input"]
                }
        #print('data["input"][v]', data["input"][variables[0]].shape, flush=True)
        x = np.stack([data["input"][v] for v in variables], axis=0)
        return x

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.means) / self.stds

    def _unstandardize(self, x: np.ndarray) -> np.ndarray:
        return x * self.stds + self.means

    def __len__(self) -> int:
        return len(self.files[: -self.interval])  # last files dont have a target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = time.time()
        loaded = self._load_file(self.files[idx], self.variables)
        end = time.time()
        #print("dl time", end-start, self.files[idx], flush=True)
        start = time.time()
        x = torch.from_numpy(
            self._standardize(loaded)
        ).float()
        end = time.time()
        #print("_standardize time", end-start, flush=True)
        t = torch.from_numpy(
            self._standardize(
                self._load_file(self.files[idx + self.interval], self.variables)
            )
        ).float()
        return x, t  # C x H x W

class ERA5NRMRTIDataset(Dataset):
    #Non-redisual monolithic (preprocessed single file) random time interval dataset
    def __init__(
        self,
        root = '/flare/datasets/wb2/0.25deg_1_step_6hr_h5df_fix_bug/',
        path = '/flare/datasets/wb2/testing/monolithic_striped/monolithic2.hdf5'
    ):
        super().__init__()
        self.root = root
        self.path = path

        self.max_interval = 4
        self.intervals = [6,12,24]
        self.shape = [69,721,1440]
        self.read = 0

    @property
    def n_channels(self):
        assert len(self.shape) == 3
        return self.shape[0]

    @property
    def img_resolution(self):
        return self.shape[1], self.shape[2]

    @property
    def variables(self):
        return ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200", "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500", "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "geopotential_1000", "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "u_component_of_wind_1000", "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "v_component_of_wind_1000", "temperature_50", "temperature_100", "temperature_150", "temperature_200", "temperature_250", "temperature_300", "temperature_400", "temperature_500", "temperature_600", "temperature_700", "temperature_850", "temperature_925", "temperature_1000", "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200", "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500", "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925", "specific_humidity_1000"]

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.load(os.path.join(self.root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(self.root, "lon.npy")).astype(np.float32)
        return lat, lon
    
    def _load_and_stack(self, filename: str) -> np.ndarray:
        root = self.root
        with np.load(os.path.join(root, filename)) as data:
            return np.stack([data[v] for v in self.variables], axis=0).reshape(-1, 1, 1)
    
    def _setup_standardize(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.x_means = self._load_and_stack("normalize_mean.npz")
        self.x_stds = self._load_and_stack("normalize_std.npz")

        #self.t_means = np.zeros_like(self.x_means)
        self.t_stds = {t:self._load_and_stack(f"normalize_diff_std_{t}.npz") for t in self.intervals}
        self.t_means = {t:self._load_and_stack(f"normalize_diff_mean_{t}.npz") for t in self.intervals}

    def standardize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_means) / self.x_stds

    def unstandardize_x(self, x: np.ndarray) -> np.ndarray:
        return x * self.x_stds + self.x_means

    def standardize_t(self, t: np.ndarray, dt) -> np.ndarray:
        return (t - self.t_means[dt]) / self.t_stds[dt]

    def unstandardize_t(self, t: np.ndarray, dt) -> np.ndarray:
        return t * self.t_stds[dt] + self.t_means[dt]
    
    def get_time(self, idx: int) -> np.datetime64:
        with h5py.File(self.files[idx], "r") as f:
            timestamp = f["input"]["time"][()]
        return np.datetime64(timestamp.decode("utf-8"))

    def __len__(self) -> int:
        return (58440-4)*3  # last files dont have a target

    def open_file(self, wp_y_rank, wp_x_rank, dtype=torch.float32, load_x=True, load_t=True, benchmark=False, data_shape=None):
        self.wp_y_rank = wp_y_rank
        self.wp_x_rank = wp_x_rank
        self.load_t = load_t
        #self.intervals = intervals
        path = os.path.join(self.path, f"y_{wp_y_rank}_x_{wp_x_rank}.hdf5")

        if benchmark:
            self._setup_standardize()
            #self.ds = self.ds[:104][()]
            self.benchmark = benchmark
            self.ds = torch.randn(data_shape)
        else:
            self.file = h5py.File(path, 'r')
            self.ds = self.file["input"]
            self.benchmark = benchmark
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.read > 10:
            gc.collect()
            self.read = 0
        if self.benchmark:
            idx = idx%25
        interval = idx%3 #interval index from interval list
        interval = self.intervals[interval] #interval in hours
        dt = interval//6 #interval in dataset indices

        x_ind = idx//3
        t_ind = x_ind+dt
        
        dt = torch.tensor([dt],dtype=torch.float32)

        x = np.array(self.ds[x_ind,:,:,:])
        if self.benchmark:
            if not self.load_t:
                return x, dt
            else:
                return x, x, dt
        self.read += 1
        if not self.load_t:
            assert len(x.shape) == 3, (x.shape, self.ds.shape, x_ind, interval, dt)# ((2, 69, 180, 360), 20298, 6, tensor([1.]))
            x = torch.from_numpy(self.standardize_x(x)).float()
            assert len(x.shape) == 3, x.shape
            return x, dt
        
        t = np.array(self.ds[t_ind,:,:,:])
        t = t - x

        x = torch.from_numpy(self.standardize_x(x)).float()
        t = torch.from_numpy(self.standardize_t(t, interval)).float()

        return x, t, dt  # C x H x W


class ERA5NRMDataset(Dataset):
    #Non-redisual monolithic (preprocessed single file) dataset
    def __init__(
        self,
        root = '/flare/datasets/wb2/0.25deg_1_step_6hr_h5df_fix_bug/',
        path = '/flare/datasets/wb2/testing/monolithic_striped/monolithic2.hdf5',
        enhanced_channels3_train = False,
        enhanced_channels4_train = False,
        interval = 6,
        test = False,
        enhanced_channels4_inference = False,
    ):
        super().__init__()
        self.root = root
        self.path = path
        self.interval = interval
        self.samples = (1464-(interval//6)) if test else (58440-(interval//6))
        self.enhanced_channels = enhanced_channels3_train or enhanced_channels4_train or enhanced_channels4_inference
        self.enhanced_channels3_train = enhanced_channels3_train
        self.enhanced_channels4_train = enhanced_channels4_train
        self.enhanced_channels4_inference = enhanced_channels4_inference
        if self.enhanced_channels4_train or self.enhanced_channels4_inference:
            assert interval==24, "Sea surface temperature only implemented for 24h prediction"
            self.channels = 73
        elif self.enhanced_channels:
            self.channels = 72
        else:
            self.channels = 69
        self.shape = [self.channels,721,1440]
        self.read = 0
        self.sst_mask = None
        self._setup_standardize()

    @property
    def n_channels(self):
        assert len(self.shape) == 3
        return self.shape[0]

    @property
    def img_resolution(self):
        return self.shape[1], self.shape[2]

    @property
    def variables(self):
        if self.enhanced_channels4_train or self.enhanced_channels4_inference:
            variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200", "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500", "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "geopotential_1000", "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "u_component_of_wind_1000", "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "v_component_of_wind_1000", "temperature_50", "temperature_100", "temperature_150", "temperature_200", "temperature_250", "temperature_300", "temperature_400", "temperature_500", "temperature_600", "temperature_700", "temperature_850", "temperature_925", "temperature_1000", "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200", "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500", "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925", "specific_humidity_1000", "sea_surface_temperature" ,"toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask"]
        elif self.enhanced_channels:
            variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200", "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500", "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "geopotential_1000", "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "u_component_of_wind_1000", "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "v_component_of_wind_1000", "temperature_50", "temperature_100", "temperature_150", "temperature_200", "temperature_250", "temperature_300", "temperature_400", "temperature_500", "temperature_600", "temperature_700", "temperature_850", "temperature_925", "temperature_1000", "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200", "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500", "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925", "specific_humidity_1000", "toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask"]
        else:
            variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200", "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500", "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "geopotential_1000", "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "u_component_of_wind_1000", "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "v_component_of_wind_1000", "temperature_50", "temperature_100", "temperature_150", "temperature_200", "temperature_250", "temperature_300", "temperature_400", "temperature_500", "temperature_600", "temperature_700", "temperature_850", "temperature_925", "temperature_1000", "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200", "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500", "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925", "specific_humidity_1000"]
        return variables

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.load(os.path.join(self.root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(self.root, "lon.npy")).astype(np.float32)
        return lat, lon
    
    def _load_and_stack(self, filename: str) -> np.ndarray:
        root = self.root
        with np.load(os.path.join(root, filename)) as data:
            return np.stack([data[v] for v in self.variables], axis=0).reshape(-1, 1, 1)
    
    def _setup_standardize(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.x_means = self._load_and_stack("normalize_mean.npz")
        self.x_stds = self._load_and_stack("normalize_std.npz")

        #self.t_means = np.zeros_like(self.x_means)
        self.t_stds = self._load_and_stack(f"normalize_diff_std_{self.interval}.npz")
        self.t_means = self._load_and_stack(f"normalize_diff_mean_{self.interval}.npz")

    def standardize_x(self, x: np.ndarray, start_ch=0, end_ch=None, clamp_sst=False) -> np.ndarray:
        out = x.copy()
        if clamp_sst and self.sst_mask is not None:
            sst_ind = self.variables.index("sea_surface_temperature")
            #out[sst_ind][self.sst_mask] = np.nanmin(out[sst_ind])
            np.copyto(out[sst_ind], self.sst_min, where=self.sst_mask)
        out = (x - self.x_means[start_ch:end_ch]) / self.x_stds[start_ch:end_ch]
        return out

    def unstandardize_x(self, x: np.ndarray, start_ch=0, end_ch=None) -> np.ndarray:
        return x * self.x_stds[start_ch:end_ch] + self.x_means[start_ch:end_ch]

    def standardize_t(self, t: np.ndarray, start_ch=0, end_ch=None) -> np.ndarray:
        return (t - self.t_means[start_ch:end_ch]) / self.t_stds[start_ch:end_ch]

    def unstandardize_t(self, t: np.ndarray, start_ch=0, end_ch=None) -> np.ndarray:
        return t * self.t_stds[start_ch:end_ch] + self.t_means[start_ch:end_ch]
    
    def get_time(self, idx: int) -> np.datetime64:
        with h5py.File(self.files[idx], "r") as f:
            timestamp = f["input"]["time"][()]
        return np.datetime64(timestamp.decode("utf-8"))

    def __len__(self) -> int:
        return self.samples  # last files dont have a target

    def open_file(self, wp_y_rank, wp_x_rank, dtype=torch.float32, load_x=True, load_t=True, load_inf_labels=0, benchmark=False):
        self.wp_y_rank = wp_y_rank
        self.wp_x_rank = wp_x_rank
        self.load_t = load_t
        self.load_inf_labels = load_inf_labels
        #self.intervals = intervals
        path = os.path.join(self.path, f"y_{wp_y_rank}_x_{wp_x_rank}.hdf5")

        self.file = h5py.File(path, 'r')
        self.ds = self.file["input"]
        self.benchmark = benchmark
        if self.benchmark:
            self.ds = self.ds[:104][()]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.read > 10:
            gc.collect()
            self.read = 0
        if self.benchmark:
            idx = idx%100

        x_ind = idx
        dt = (self.interval//6)
        t_ind = x_ind+dt
        
        x = np.array(self.ds[x_ind,:,:,:])
        self.read += 1
        if self.enhanced_channels3_train or self.enhanced_channels4_train:
            rad_ind = self.variables.index("toa_incident_solar_radiation")
            t = np.array(self.ds[t_ind,:rad_ind+1,:,:])
            if self.enhanced_channels4_train:
                sst_ind = self.variables.index("sea_surface_temperature")
                np.copyto(t[sst_ind], np.nanmin(t[sst_ind]), where=np.isnan(t)[sst_ind])
                np.copyto(x[sst_ind], np.nanmin(x[sst_ind]), where=np.isnan(x)[sst_ind])
            extra_dim = t[rad_ind:rad_ind+1,:,:]
            x_un = x
            x = torch.from_numpy(self.standardize_x(x)).float()
            extra_dim = torch.from_numpy(self.standardize_x(extra_dim,rad_ind,rad_ind+1)).float()
            x = np.concatenate([x,extra_dim],axis=0)

            if self.load_inf_labels > 0:
                labels = np.array(self.ds[x_ind:x_ind+(self.load_inf_labels+1)*dt:dt,:,:,:])
                return x, labels
            else:
                t = t[:rad_ind,:,:]
                t = t - x_un[:rad_ind]
                t = torch.from_numpy(self.standardize_t(t, 0, rad_ind)).float()
                return x, t  # C x H x W
        
        #Labels needed for local eval
        if self.load_inf_labels > 0:
            if self.enhanced_channels4_inference:
                sst_ind = self.variables.index("sea_surface_temperature")
                if self.sst_mask is None:
                    self.sst_mask = np.isnan(x[sst_ind])
                    self.sst_min = np.nanmin(x[sst_ind])
                np.copyto(x[sst_ind], np.nanmin(x[sst_ind]), where=np.isnan(x)[sst_ind])
            x = torch.from_numpy(self.standardize_x(x)).float()
            labels = np.array(self.ds[x_ind:x_ind+(self.load_inf_labels+1)*dt:dt,:,:,:])
            if self.enhanced_channels4_inference:
                np.copyto(labels[:,sst_ind], np.nanmin(labels[:,sst_ind]), where=np.isnan(labels[:,sst_ind]))
            return x, labels

        if not self.load_t:
            assert len(x.shape) == 3, (x.shape, self.ds.shape, x_ind)# ((2, 69, 180, 360), 20298, 6, tensor([1.]))
            x = torch.from_numpy(self.standardize_x(x)).float()
            assert len(x.shape) == 3, x.shape
            return x
        
        t = np.array(self.ds[t_ind,:,:,:])
        t = t - x

        x = torch.from_numpy(self.standardize_x(x)).float()
        t = torch.from_numpy(self.standardize_t(t)).float()

        return x, t  # C x H x W