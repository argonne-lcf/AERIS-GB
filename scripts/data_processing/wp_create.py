# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.

# AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions
# This work is licensed under the MIT License. See LICENSE for details.

from mpi4py import MPI

import h5py
import numpy as np
import time
from einops import rearrange


def generate_window_data_mask(window_size, y_rank, x_rank, WP_Y, WP_X, inputs: np.ndarray):
    #mask = np.array(np.zeros(shape.numel()), dtype='bool')
    ws_y = window_size[0]
    ws_x = window_size[1]
    print("ws_y, ws_x, WP_Y, WP_X",ws_y, ws_x, WP_Y, WP_X,flush=True)
    #indices = np.arange(shape.numel()).reshape(shape)
    indices = np.arange(inputs.size).reshape(inputs.shape)
    indices = indices[:,1:,:]
    indices = rearrange(indices, "c (wc_y ws_y) (wc_x ws_x) -> c wc_y ws_y wc_x ws_x", ws_y=ws_y, ws_x=ws_x)
    #Pick the correct windows:
    indices = indices[:,y_rank::WP_Y,:,x_rank::WP_X,:]
    #indices = rearrange(indices, "b c wc_y ws_y wc_x ws_x -> (b wc_y wc_x) c ws_y ws_x")
    indices = rearrange(indices, "c wc_y ws_y wc_x ws_x ->c (wc_y ws_y) (wc_x ws_x)")
    loaded_shape = indices.shape
    indices = indices.flatten()
    mask = np.zeros(inputs.size, dtype='bool')
    mask[indices] = True
    return mask.reshape(inputs.shape),[loaded_shape[1],loaded_shape[2]]

if __name__ == "__main__":
    rank = int(MPI.COMM_WORLD.Get_rank())
    world_size = int(MPI.COMM_WORLD.Get_size())

    MPI.COMM_WORLD.barrier()
    filename = "/flare/SAFS/vhat/data/enchanced_merged_test.hdf5"
    path = "/flare/SAFS/vhat/data/nonres-60x60-enchanced-test/y_4_x_4/"
    channels = 72
    
    samples = 1464
    #samples = 58440
    
    read_f = h5py.File(filename, 'r')
    read_ds = read_f["input"]
    WP_Y = 4
    WP_X = 4
    WS = (60,60)
    DP=world_size//(WP_Y*WP_X)
    grid = np.arange(DP*WP_Y*WP_X).reshape(DP,WP_Y,WP_X)#Y,X

    my_coords = np.where(grid==rank)
    
    x_rank = my_coords[2].item()
    y_rank = my_coords[1].item()
    dp_rank = my_coords[0].item()
    #path = "/flare/datascience/vhat/fix3/y_2_x_2/"
    overwrite = True
    if overwrite and dp_rank==0:
        write_f = h5py.File(path+f"y_{y_rank}_x_{x_rank}.hdf5", 'w')
        write_ds = write_f.create_dataset("input", (samples,channels,720//WP_Y,1440//WP_X), 'f')
        write_f.close()
    MPI.COMM_WORLD.Barrier()
    write_f = h5py.File(path+f"y_{y_rank}_x_{x_rank}.hdf5", 'a')
    write_ds = write_f["input"]
    #exit()
    mask,loaded_shape = generate_window_data_mask(WS, y_rank, x_rank, WP_Y, WP_X, np.zeros((channels,721,1440)))

    assert loaded_shape[0] == (720//WP_Y) and loaded_shape[1] == (1440//WP_X), f"{loaded_shape[0]}, {loaded_shape[1]}"
    #58440-1
    start = 0
    stop = samples
    #stop = 58440-1
    #stop = 30000
    #start = 55000
    items = 0
    for i in range(start+dp_rank,stop,DP):
        start = time.time()
        read = np.array(read_ds[i,:,:,:][mask][()]).reshape((channels, loaded_shape[0], loaded_shape[1]))
        if channels == 73:
            assert read[:-4].var() > 0.01 and read[-3:].var() > 0.01, f"read zeros for some reason at {i}, {read[:-1].shape}"
        else:
            assert read.var() > 0.01, f"read zeros for some reason at {i}"

        write_ds[i,:,:,:] = read
        end = time.time()
        print(f"{i} in {end-start}", read.var(), flush=True)
        if items == 100:
            write_f.close()
            write_f = h5py.File(path+f"y_{y_rank}_x_{x_rank}.hdf5", 'a')
            write_ds = write_f["input"]
            items = 0
        items += 1
        
        
    read_f.close()
    write_f.close()
    print("done")

