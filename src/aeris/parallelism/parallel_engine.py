# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.

# AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions
# This work is licensed under the MIT License. See LICENSE for details.

from mpi4py import MPI
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from deepspeed.runtime.pipe import schedule
from aeris.parallelism.schedule import LessCustomTrainSchedule, SubmitRecvGrad, SubmitRecvActivation, ProcessActivation, ProcessGrad

from types import MethodType
import copy
import math
import time
import os
import gc

def parallel_TrigFlow_calculate_noise(x, rng, nl_rng, sigma_min, sigma_max, sigma_data, dtype=torch.float32):
    def loguniform(x: torch.Tensor, sigma_min: float, sigma_max: float, rng) -> torch.Tensor:
        sigma_min = torch.tensor(sigma_min, device=x.device)
        sigma_max = torch.tensor(sigma_max, device=x.device)
        u = torch.rand([1,1], device=x.device, generator=nl_rng)#[b,d]
        us = torch.log(sigma_min) + u * (torch.log(sigma_max) - torch.log(sigma_min))
        return torch.exp(us)

    tau = loguniform(x, sigma_min, sigma_max, rng)
    t = torch.atan(tau / sigma_data)  # [0, pi/2]
    z = torch.randn(x.shape, dtype=dtype, device=x.device, generator=rng) * sigma_data

    return t, z


def wait_handle(handle):
    if handle != None:
        handle.wait()
        handle = None

class ParallelEngine():
    def __init__(
            self,
            cfg,
            PP,
            SP,
            WP_X,
            WP_Y,
            GAS,
            rank,
            world_size,
            seq,
            batch_size,
            device,
            init_model_fn,
            sigma_data = 1.0,
            checkpoint_path: str = "/flare/Aurora_deployment/vhat/gb25_cli/aeris/checkpoints/trash",  # min factor relative to lr [0,1].
            data_dtype = torch.float32,
            model_dtype = torch.bfloat16,
            total_img: int = 1_000_000,
            ema_halflife_img: int = 500_000,  # half-life of EMA of model weights.
            ema_rampup_ratio: float = 0.05,  # EMA ramp-up coefficient, None = disable.
            lr_rampup_img: int = 40_000,  # n learning rate ramp-up.
            lr_min_factor: float = 0.01,  # min factor relative to lr [0,1].
            **init_model_kwargs
    ):
        #Way too many args. Refactoring required.
        self.rank = rank
        self.world_size = world_size
        self.PP = PP
        self.SP = SP
        assert SP== 12, "No ParallelEngine implementation for SP!=12"
        self.WP_X = WP_X
        self.WP_Y = WP_Y
        self.GAS = GAS
        self.device = device
        self.window_size = init_model_kwargs["window_size"]
        self.dim = init_model_kwargs["dim"]
        self.ema_halflife_img = ema_halflife_img
        self.ema_rampup_ratio = ema_rampup_ratio
        self.lr_rampup_img = lr_rampup_img
        self.lr_min_factor = lr_min_factor
        self.total_img = total_img
        self.base_lr = None
        self.base_lr_num = cfg.optimizer.lr
        self.checkpoint_path = checkpoint_path
        self.sigma_data = sigma_data
        self.sigma_min= cfg.model.sigma_min
        self.sigma_max = cfg.model.sigma_max
        self.rit = init_model_kwargs["rit"]
        self.cfg = cfg
        self.w_lat_add = cfg.model.w_lat_add
        self.grad_clip = cfg.model.grad_clip
        self.grad_clip_norm_type = cfg.model.grad_clip_norm_type
        self.grad_clip_max_norm = cfg.model.grad_clip_max_norm
        self.dist_optim = cfg.model.dist_optimizer
        self.benchmark = cfg.model.benchmark
        self.gpu_datamovement = cfg.model.gpu_datamovement
        self.rs_grads = cfg.model.rs_grads
        self.diffusion = cfg.model.diffusion
        self.layerwise_t_emb = cfg.model.layerwise_t_emb
        self.condition_channels = init_model_kwargs["condition_channels"]
        self.input_channels = init_model_kwargs["input_channels"]
        self.model_in_channels = init_model_kwargs["model_in_channels"]
        self.model_out_channels = init_model_kwargs["model_out_channels"]

        self.overlap_p2p_comms = cfg.model.overlap_p2p_comms
        if self.overlap_p2p_comms:
            self.comm = MPI.COMM_WORLD
            assert self.rit, "overlapped p2p comms only enabled for random interval training"

        self.noise_rng_seed = 0
        if self.rank==0:
            print("start grid init", flush=True)
        self.grid_init()
        if self.rank==0:
            print("done grid init", flush=True)
        self.setup_schedule()
        self.setup_noise_rng(seed=self.noise_rng_seed+self.wp_x_rank*self.WP_Y*self.DP*self.SP+self.wp_y_rank*self.DP*self.SP+self.dp_rank*self.SP+self.sp_rank)
        self.setup_noise_level_rng(seed=self.noise_rng_seed+self.dp_rank)
        self.setup_buffers(init_model_kwargs["dim"], seq, batch_size)

        self.data_dtype = data_dtype
        self.model_dtype = model_dtype
        self.local_model_dtype = torch.float32 if self.data_dtype==torch.float32 and (self.is_first_stage() or self.is_last_stage()) else torch.bfloat16 
        if self.dist_optim:
            self.grad_dtype = torch.bfloat16
            assert cfg.model.custom_decay == False, "param groups not implemented for distributed optimizer"
            self.model = init_model_fn(self, cfg, **init_model_kwargs).to(device=device, dtype=self.local_model_dtype)
            self.init_params(self.model, init_model_kwargs["dim"], cfg)
            self.setup_grad_acc_distopt()
            self.main_model = self.local_param_buf_opt
            #self.init_params(init_model_kwargs["dim"], cfg)
            #self.model = copy.deepcopy(self.main_model).to(device=self.device, dtype=self.local_model_dtype)
            self.ema = None #Not implemented
        else:
            self.main_model = init_model_fn(self, cfg, **init_model_kwargs).to(device="cpu", dtype=torch.float32)
            self.init_params(self.main_model, init_model_kwargs["dim"], cfg)
            self.model = copy.deepcopy(self.main_model).to(device=self.device, dtype=self.local_model_dtype)
            self.setup_grad_acc_locopt()
            self.ema = copy.deepcopy(self.main_model).to(device="cpu").eval().requires_grad_(False)

        self.loss = 0.0
        self.total_norm = 0.0
        self.stage_norm = 0.0
        self.tot_loss = 0.0
        self.cur_nimg = 0
        #self.samples_loaded = 0
    def setup_dataset(self,dataset, cfg):
        if self.is_first_stage() or self.is_last_stage():
            self.dataset=dataset
            self.wp_load = cfg.data.wp_load
            if self.rank == 0:
                print("opening dataset", flush=True)
            if self.diffusion and self.sp_rank==0:
                dataset.open_file(self.wp_y_rank, self.wp_x_rank, load_x=True, load_t=True, benchmark=self.benchmark)
            elif cfg.model.rit and self.sp_rank==0:
                dataset.open_file(self.wp_y_rank, self.wp_x_rank, load_x=self.is_first_stage(), load_t=self.is_last_stage(), benchmark=self.benchmark, data_shape=(30,69,720//self.WP_Y,1440//self.WP_X))
            elif cfg.data.wp_load and self.sp_rank==0:
                dataset.open_file(self.wp_y_rank, self.wp_x_rank, load_x=self.is_first_stage(), load_t=self.is_last_stage())
            if not cfg.data.wp_load and not cfg.model.rit:
                raise NotImplementedError
                #TODO all below needs double checking the residuals
                res = dataset.img_resolution
                assert res == (721,1440), "other resolutions not implemented"
                shape = np.zeros((1, dataset.n_channels, 721, 1440))
                ind_mask, loaded_shape = self.generate_window_data_mask(shape)
                self.dataset.set_mask(ind_mask[:,:,:,:], loaded_shape)

    def setup_data_sampling(self, dataset_sampler, dataloader):
        self.dataset_sampler = dataset_sampler
        self.dataloader = dataloader
        
        
    def wp_dataset_load_diffusion(self, dtype):
        ws_y, ws_x = self.window_size
        wc_y_tot = 720//ws_y
        wc_x_tot = 1440//ws_x
        wc_y_loc = (wc_y_tot//self.WP_Y)
        wc_x_loc = (wc_x_tot//self.WP_X)
        wc_local = wc_y_loc*wc_x_loc

        input_res = torch.zeros((wc_local, ws_y*ws_x//self.SP, self.condition_channels), dtype=dtype)
        label_res = torch.zeros((wc_local, ws_y*ws_x//self.SP, self.input_channels), dtype=dtype)

        if self.gpu_datamovement:
            input_res = input_res.to(self.device)
            label_res = label_res.to(self.device)
        
        if self.sp_rank == 0:
            #start = time.time()
            s = self.SP//4
            inputs, labels = next(self.dataloader)

            if self.is_first_stage():
                inputs = inputs.to(dtype)
                inputs = rearrange(inputs, "b c (wc_y d ws_y_h) (wc_x e ws_x_h) -> (b wc_y wc_x) c (d e) (ws_y_h ws_x_h)", d=2, e=2, ws_y_h=ws_y//2, ws_x_h=ws_x//2)
                inputs = rearrange(inputs, "b c w (s n) -> b (w s) n c", s=s)# n is tokens left from window after SP division. normally here we have:(b wc_y wc_x) (ws_y  ws_x) c
                input_list = [item[:,0,:,:].contiguous() for item in inputs.chunk(self.SP, dim=1)]
                if self.gpu_datamovement:
                    input_list = [item[:,0,:,:].contiguous().to(self.device) for item in inputs.chunk(self.SP, dim=1)]
                else:
                    input_list = [item[:,0,:,:].contiguous() for item in inputs.chunk(self.SP, dim=1)]

                torch.distributed.scatter(input_res, scatter_list=input_list, src=self.rank,group=self.sp_group)

            labels = labels.to(dtype)[:,:self.input_channels]
            labels = rearrange(labels, "b c (wc_y d ws_y_h) (wc_x e ws_x_h) -> (b wc_y wc_x) c (d e) (ws_y_h ws_x_h)", d=2, e=2, ws_y_h=ws_y//2, ws_x_h=ws_x//2)
            labels = rearrange(labels, "b c w (s n) -> b (w s) n c", s=s) # n is tokens left from window after SP division. normally here we have:(b wc_y wc_x) (ws_y  ws_x) c
            if self.gpu_datamovement:
                labels_list = [item[:,0,:,:].contiguous().to(self.device) for item in labels.chunk(self.SP, dim=1)] 
            else:
                labels_list = [item[:,0,:,:].contiguous() for item in labels.chunk(self.SP, dim=1)] 
            torch.distributed.scatter(label_res, scatter_list=labels_list, src=self.rank,group=self.sp_group)
        else:
            if self.is_first_stage():
                torch.distributed.scatter(input_res, scatter_list=None, src=self.grid[self.dp_rank,self.wp_y_rank,self.wp_x_rank,self.pp_rank,0],group=self.sp_group)
            torch.distributed.scatter(label_res, scatter_list=None, src=self.grid[self.dp_rank,self.wp_y_rank,self.wp_x_rank,self.pp_rank,0],group=self.sp_group)
        
        return input_res, label_res
    
    def wp_dataset_load(self, dtype):
        ws_y, ws_x = self.window_size
        wc_y_tot = 720//ws_y
        wc_x_tot = 1440//ws_x
        wc_y_loc = (wc_y_tot//self.WP_Y)
        wc_x_loc = (wc_x_tot//self.WP_X)
        wc_local = wc_y_loc*wc_x_loc

        input_res = torch.zeros((wc_local, ws_y*ws_x//self.SP, 69), dtype=dtype)
        label_res = torch.zeros((wc_local, ws_y*ws_x//self.SP, 69), dtype=dtype)
        interval = torch.zeros((1,1), dtype=dtype)
        if self.gpu_datamovement:
            input_res = input_res.to(self.device)
        if self.gpu_datamovement:
            label_res = label_res.to(self.device)
        
        if self.sp_rank == 0:
            #start = time.time()
            s = self.SP//4
            if self.is_first_stage():
                inputs = next(self.dataloader)
                if self.rit:
                    inputs, interval = inputs
                    assert len(inputs.shape) == 4, inputs.shape
                inputs = inputs.to(dtype)
                #end = time.time()
                #report_timers = False
                #if report_timers and self.rank==0:
                #    print(self.rank, "next(self.dataloader)", end-start, flush=True)
                #print(self.rank, "loaded data from dataloader", inputs.shape)
                #print(self.rank, "dataloader_group", torch.distributed.get_process_group_ranks(self.dataloader_group))
                #b: batch, c: channels, wc: window count, ws_(h): window size (half), d/e constant two, 
                inputs = rearrange(inputs, "b c (wc_y d ws_y_h) (wc_x e ws_x_h) -> (b wc_y wc_x) c (d e) (ws_y_h ws_x_h)", d=2, e=2, ws_y_h=ws_y//2, ws_x_h=ws_x//2)
                inputs = rearrange(inputs, "b c w (s n) -> b (w s) n c", s=s)# n is tokens left from window after SP division. normally here we have:(b wc_y wc_x) (ws_y  ws_x) c
                input_list = [item[:,0,:,:].contiguous() for item in inputs.chunk(self.SP, dim=1)]
                if self.gpu_datamovement:
                    input_list = [item[:,0,:,:].contiguous().to(self.device) for item in inputs.chunk(self.SP, dim=1)]
                else:
                    input_list = [item[:,0,:,:].contiguous() for item in inputs.chunk(self.SP, dim=1)]

                torch.distributed.scatter(input_res, scatter_list=input_list, src=self.rank,group=self.sp_group)

            if self.is_last_stage():
                labels = next(self.dataloader)
                if self.rit:
                    inputs, labels, interval = labels
                labels = labels.to(dtype)
                labels = rearrange(labels, "b c (wc_y d ws_y_h) (wc_x e ws_x_h) -> (b wc_y wc_x) c (d e) (ws_y_h ws_x_h)", d=2, e=2, ws_y_h=ws_y//2, ws_x_h=ws_x//2)
                labels = rearrange(labels, "b c w (s n) -> b (w s) n c", s=s) # n is tokens left from window after SP division. normally here we have:(b wc_y wc_x) (ws_y  ws_x) c
                if self.gpu_datamovement:
                    labels_list = [item[:,0,:,:].contiguous().to(self.device) for item in labels.chunk(self.SP, dim=1)] 
                else:
                    labels_list = [item[:,0,:,:].contiguous() for item in labels.chunk(self.SP, dim=1)] 
                torch.distributed.scatter(label_res, scatter_list=labels_list, src=self.rank,group=self.sp_group)
        else:
            if self.is_first_stage():
                torch.distributed.scatter(input_res, scatter_list=None, src=self.grid[self.dp_rank,self.wp_y_rank,self.wp_x_rank,self.pp_rank,0],group=self.sp_group)
            if self.is_last_stage():
                torch.distributed.scatter(label_res, scatter_list=None, src=self.grid[self.dp_rank,self.wp_y_rank,self.wp_x_rank,self.pp_rank,0],group=self.sp_group)
        
        if self.rit:
            if self.gpu_datamovement:
                interval = interval.to(self.device)
            time.sleep(0.001)#I do not trust oneCCL scatter because gather had issues too.
            torch.distributed.broadcast(interval, src=self.grid[self.dp_rank,self.wp_y_rank,self.wp_x_rank,self.pp_rank,0],group=self.sp_group)
            return input_res, label_res, interval 

        
        #print(self.rank, "done broadcasting", flush=True)
        
        return input_res, label_res

    def grid_init(self):
        WP_Y = self.WP_Y
        WP_X = self.WP_X
        PP = self.PP
        SP = self.SP
        assert WP_Y*WP_X*PP*SP <= self.world_size, f"Not enough ranks to place parallelisms to {WP_Y},{WP_X},{PP},{SP} -> {self.world_size}"
        DP = self.world_size//(WP_Y*WP_X*PP*SP)
        self.DP = DP
        self.grid = np.arange(DP*WP_Y*WP_X*PP*SP).reshape((DP,WP_Y,WP_X,PP,SP))
        #print("grid.shape", self.grid.shape, DP)
        self.my_coords = np.where(self.grid==self.rank)
        self.sp_rank = self.my_coords[4][0].item()
        self.pp_rank = self.my_coords[3][0].item()
        self.wp_x_rank = self.my_coords[2][0].item()
        self.wp_y_rank = self.my_coords[1][0].item()
        self.dp_rank = self.my_coords[0][0].item()

        me = self.grid[self.dp_rank, self.wp_y_rank, self.wp_x_rank, self.pp_rank, self.sp_rank]
        assert me == self.rank

        self.shift_up_next = self.pp_rank%2 == 1


        #Constructed from a grid:
        #X order:
        #0,1
        #0,1
        #Y order:
        #0,0
        #1,1
        """
        0,1,2 | 3,4,5   || 0,1,2 | 3,4,5
        ----------------------------------
        6,7,8 | 9,10,11 || 6,7,8 | 9,10,11
        ----------------------------------
        ----------------------------------
        0,1,2 | 3,4,5   || 0,1,2 | 3,4,5
        ----------------------------------
        6,7,8 | 9,10,11 || 6,7,8 | 9,10,11
        
        """
        if self.shift_up_next:
            if self.sp_rank<3:
                self.next_rank = self.grid[self.dp_rank, (self.wp_y_rank-1)%WP_Y, (self.wp_x_rank-1)%WP_X,  (self.pp_rank+1)%PP, self.sp_rank+9]
                self.prev_rank = self.grid[self.dp_rank, (self.wp_y_rank-1)%WP_Y, (self.wp_x_rank-1)%WP_X,  (self.pp_rank-1)%PP, self.sp_rank+9]
            elif self.sp_rank<6:
                self.next_rank = self.grid[self.dp_rank, (self.wp_y_rank-1)%WP_Y,           self.wp_x_rank, (self.pp_rank+1)%PP, self.sp_rank+3]
                self.prev_rank = self.grid[self.dp_rank, (self.wp_y_rank-1)%WP_Y,           self.wp_x_rank, (self.pp_rank-1)%PP, self.sp_rank+3]
            elif self.sp_rank<9:
                self.next_rank = self.grid[self.dp_rank, self.wp_y_rank,        (self.wp_x_rank-1)%WP_X,    (self.pp_rank+1)%PP, self.sp_rank-3]
                self.prev_rank = self.grid[self.dp_rank, self.wp_y_rank,        (self.wp_x_rank-1)%WP_X,    (self.pp_rank-1)%PP, self.sp_rank-3]
            elif self.sp_rank<12:
                self.next_rank = self.grid[self.dp_rank, self.wp_y_rank,                    self.wp_x_rank, (self.pp_rank+1)%PP, self.sp_rank-9]
                self.prev_rank = self.grid[self.dp_rank, self.wp_y_rank,                    self.wp_x_rank, (self.pp_rank-1)%PP, self.sp_rank-9]
            else:
                raise NotImplementedError
        else:
            if self.sp_rank<3:
                self.next_rank = self.grid[self.dp_rank, self.wp_y_rank,                        self.wp_x_rank, (self.pp_rank+1)%PP, self.sp_rank+9]
                self.prev_rank = self.grid[self.dp_rank, self.wp_y_rank,                        self.wp_x_rank, (self.pp_rank-1)%PP, self.sp_rank+9]
            elif self.sp_rank<6:
                self.next_rank = self.grid[self.dp_rank, self.wp_y_rank,            (self.wp_x_rank+1)%WP_X,    (self.pp_rank+1)%PP, self.sp_rank+3]
                self.prev_rank = self.grid[self.dp_rank, self.wp_y_rank,            (self.wp_x_rank+1)%WP_X,    (self.pp_rank-1)%PP, self.sp_rank+3]
            elif self.sp_rank<9:
                self.next_rank = self.grid[self.dp_rank, (self.wp_y_rank+1)%WP_Y,               self.wp_x_rank, (self.pp_rank+1)%PP, self.sp_rank-3]
                self.prev_rank = self.grid[self.dp_rank, (self.wp_y_rank+1)%WP_Y,               self.wp_x_rank, (self.pp_rank-1)%PP, self.sp_rank-3]
            elif self.sp_rank<12:
                self.next_rank = self.grid[self.dp_rank, (self.wp_y_rank+1)%WP_Y,   (self.wp_x_rank+1)%WP_X,    (self.pp_rank+1)%PP, self.sp_rank-9]
                self.prev_rank = self.grid[self.dp_rank, (self.wp_y_rank+1)%WP_Y,   (self.wp_x_rank+1)%WP_X,    (self.pp_rank-1)%PP, self.sp_rank-9]
            else:
                raise NotImplementedError
        if self.is_last_stage():
            self.next_rank=None
        if self.is_first_stage():
            self.prev_rank=None
            #Do not shift in stage 0 to stage 1
            self.next_rank = self.grid[self.dp_rank, self.wp_y_rank,self.wp_x_rank, (self.pp_rank+1)%PP, self.sp_rank]
        if self.pp_rank == 1:
            #Do not shift in stage 0 to stage 1
            self.prev_rank = self.grid[self.dp_rank, self.wp_y_rank,self.wp_x_rank, (self.pp_rank-1)%PP, self.sp_rank]


        if self.rank==0:
            print("start sp_group init", flush=True)
        #print("me", me, "next", self.next_rank, "prev", self.prev_rank, (self.dp_rank, self.wp_y_rank, self.wp_x_rank, self.pp_rank, self.sp_rank), flush=True)
        self.sp_group = torch.distributed.new_group(self.grid[self.dp_rank, self.wp_y_rank, self.wp_x_rank, self.pp_rank, :].flatten().tolist(),use_local_synchronization=True)
        #time.sleep(0.1)
        #torch.distributed.barrier()
        #if self.rank==0:
        #    print("done sp_group init", flush=True)
        #time.sleep(0.1)
        #torch.distributed.barrier(group=self.sp_group)
        torch.distributed.all_reduce(torch.tensor(0, device=self.device),group=self.sp_group)
        torch.distributed.barrier()

        if self.rank==0:
            print("start pp_group init", flush=True)
        self.pp_group = torch.distributed.new_group(self.grid[self.dp_rank, self.wp_y_rank, self.wp_x_rank, :, self.sp_rank].flatten().tolist(),use_local_synchronization=True)
        time.sleep(0.1)
        torch.distributed.barrier()
        time.sleep(0.1)
        torch.distributed.all_reduce(torch.tensor(0, device=self.device),group=self.pp_group)
        torch.distributed.barrier()

        if self.rank==0:
            print("start weight_group init", flush=True)
        weight_ranks = self.grid[:, :, :, self.pp_rank, :].flatten().tolist()
        self.weight_group_size = len(weight_ranks)
        self.weight_group = torch.distributed.new_group(weight_ranks,use_local_synchronization=True)
        time.sleep(0.1)
        torch.distributed.barrier()
        time.sleep(0.1)
        self.weight_group_rank = torch.distributed.get_rank(self.weight_group)
        #print(self.rank, "did w group with", self.grid[:, :, :, self.pp_rank, :].flatten().tolist(), self.weight_group, flush=True)
        torch.distributed.all_reduce(torch.tensor(0, device=self.device),group=self.weight_group)
        torch.distributed.barrier()

        """
        if self.is_first_stage() or self.is_last_stage():
            dl_ranks = self.grid[self.dp_rank, :, :, self.pp_rank, :].flatten().tolist()
            self.dataloader_group = torch.distributed.new_group(dl_ranks,use_local_synchronization=True)
            torch.distributed.all_reduce(torch.tensor(0, device=self.device),group=self.dataloader_group)"""
        
        #print(self.rank, "did w group with", self.grid[:, :, :, self.pp_rank, :].flatten().tolist(), self.weight_group, flush=True)
        #torch.distributed.barrier()


        if self.rank==0:
            print("start pp_comm_group init", flush=True)
        self.pp_comm_group_next = None
        self.pp_comm_group_prev = None
        if not self.overlap_p2p_comms:

            def init_prev():
                if self.prev_rank != None:
                    self.pp_comm_group_prev = torch.distributed.new_group([self.prev_rank, self.rank],use_local_synchronization=True)
                torch.distributed.barrier()
                time.sleep(0.1)
                if self.prev_rank != None:
                    torch.distributed.all_reduce(torch.tensor(0, device=self.device),group=self.pp_comm_group_prev)
                torch.distributed.barrier()

            def init_next():
                if self.next_rank != None:
                    self.pp_comm_group_next = torch.distributed.new_group([self.rank,self.next_rank],use_local_synchronization=True)
                time.sleep(0.1)
                torch.distributed.barrier()
                if self.next_rank != None:
                    torch.distributed.all_reduce(torch.tensor(0, device=self.device),group=self.pp_comm_group_next)
                torch.distributed.barrier()
            
            #2 rounds to avoid deadlocks
            if self.pp_rank % 2 == 0:
                init_prev()
                init_next()
            else:
                init_next()
                init_prev()
    
    def is_last_stage(self):
        return self.pp_rank==self.PP-1
    
    def is_first_stage(self):
        return self.pp_rank==0

    def placeholder(self, buffer_id=None):
        pass
    
    def generate_window_data_mask(self, inputs: np.ndarray):
        #mask = np.array(np.zeros(shape.numel()), dtype='bool')
        ws_y = self.window_size[0]
        ws_x = self.window_size[1]
        assert ws_y % 2 == 0, "need to be able to divide window to two equal halves by y dimension"
        assert ws_x % 2 == 0, "need to be able to divide window to two equal halves by x dimension"
        assert ((ws_y/2)*(ws_x/2)) % 3 == 0, "Need window to be dividable by SP"
        #indices = np.arange(shape.numel()).reshape(shape)
        indices = np.arange(inputs.size).reshape(inputs.shape)
        indices = indices[:,:,1:,:]
        indices = rearrange(indices, "b c (wc_y ws_y) (wc_x ws_x) -> b c wc_y ws_y wc_x ws_x", ws_y=ws_y, ws_x=ws_x)
        #Pick the correct windows:
        indices = indices[:,:,self.wp_y_rank::self.WP_Y,:,self.wp_x_rank::self.WP_X,:]
        #indices = rearrange(indices, "b c wc_y ws_y wc_x ws_x -> (b wc_y wc_x) c ws_y ws_x")
        #Partition window for SP:
        #X order:
        #0,1
        #0,1
        #Y order:
        #0,0
        #1,1
        """
        0,1,2 | 3,4,5  
        ---------------
        6,7,8 | 9,10,11 
        
        """
        
        if self.sp_rank < 3:
            self.sp_window_start = self.sp_rank
            indices = indices[:,:,:,:ws_y//2,:,:ws_x//2]
        elif self.sp_rank < 6:
            self.sp_window_start = self.sp_rank-3
            indices = indices[:,:,:,:ws_y//2,:,ws_x//2:]
        elif self.sp_rank < 9:
            self.sp_window_start = self.sp_rank-6
            indices = indices[:,:,:,ws_y//2:,:,:ws_x//2]
        elif self.sp_rank < 12:
            self.sp_window_start = self.sp_rank-9
            indices = indices[:,:,:,ws_y//2:,:,ws_x//2:]
        else:
            raise NotImplementedError
        indices = rearrange(indices, "b c wc_y ws_y wc_x ws_x -> b c (wc_y ws_y) (wc_x ws_x)")
        loaded_shape = indices.shape
        indices = indices.flatten()
        mask = np.zeros(inputs.size, dtype='bool')
        mask[indices] = True
        return mask.reshape(inputs.shape),[loaded_shape[2],loaded_shape[3]]
    
    def slice_sp_window(self, data):

        ws_y = self.window_size[0]//2
        ws_x = self.window_size[1]//2

        #Split a quadrant into SP//3 chunks
        data = rearrange(data, "b c (wc_y ws_y) (wc_x ws_x) -> (b wc_y wc_x) (ws_y  ws_x) c", ws_y=ws_y, ws_x=ws_x) #here c is channels (hidden) and h is height. In contrast to h=heads
        #indices = rearrange(indices, "b c h w -> b (h w) c") #here c is channels (hidden) and h is height. In contrast to h=heads
        shift = data.shape[1]//3
        start = self.sp_window_start
        data = data[:,start*shift:(start+1)*shift,:]
        #mask = np.zeros(shape, dtype='bool')
        #mask[indices.flatten()] = True
        #print("done indices", flush=True)
        #print("mask.sum()",mask.sum())
        return data

    def filter_window(self, inputs: np.ndarray):
        #mask = np.array(np.zeros(shape.numel()), dtype='bool')
        ws_y = self.window_size[0]
        ws_x = self.window_size[1]
        assert ws_y % 2 == 0, "need to be able to divide window to two equal halves by y dimension"
        assert ws_x % 2 == 0, "need to be able to divide window to two equal halves by x dimension"
        assert ((ws_y/2)*(ws_x/2)) % 3 == 0, "Need window to be dividable by SP"
        #indices = np.arange(shape.numel()).reshape(shape)
        inputs = inputs[:,:,1:,:]
        inputs = rearrange(inputs, "b c (wc_y ws_y) (wc_x ws_x) -> b c wc_y ws_y wc_x ws_x", ws_y=ws_y, ws_x=ws_x)
        #Pick the correct windows:
        inputs = inputs[:,:,self.wp_y_rank::self.WP_Y,:,self.wp_x_rank::self.WP_X,:]
        #indices = rearrange(indices, "b c wc_y ws_y wc_x ws_x -> (b wc_y wc_x) c ws_y ws_x")
        #Partition window for SP:
        #X order:
        #0,1
        #0,1
        #Y order:
        #0,0
        #1,1
        """
        0,1,2 | 3,4,5  
        ---------------
        6,7,8 | 9,10,11 
        
        """
        
        if self.sp_rank < 3:
            sp_window_start = self.sp_rank
            inputs = inputs[:,:,:,:ws_y//2,:,:ws_x//2]
        elif self.sp_rank < 6:
            sp_window_start = self.sp_rank-3
            inputs = inputs[:,:,:,:ws_y//2,:,ws_x//2:]
        elif self.sp_rank < 9:
            sp_window_start = self.sp_rank-6
            inputs = inputs[:,:,:,ws_y//2:,:,:ws_x//2]
        elif self.sp_rank < 12:
            sp_window_start = self.sp_rank-9
            inputs = inputs[:,:,:,ws_y//2:,:,ws_x//2:]
        else:
            raise NotImplementedError
        inputs = rearrange(inputs, "b c wc_y ws_y wc_x ws_x -> (b wc_y wc_x) (ws_y  ws_x) c")

        ws_y = self.window_size[0]//2
        ws_x = self.window_size[1]//2

        #Split a quadrant into SP//3 chunks
        #indices = rearrange(indices, "b c h w -> b (h w) c") #here c is channels (hidden) and h is height. In contrast to h=heads
        shift = inputs.shape[1]//3
        start = sp_window_start
        inputs = inputs[:,start*shift:(start+1)*shift,:]
        #mask = np.zeros(shape, dtype='bool')
        #mask[indices.flatten()] = True
        #print("done indices", flush=True)
        #print("mask.sum()",mask.sum())
        return inputs
    
    def load(self, buffer_id):
        #inputs = self.filter_window(inputs)
        #labels = self.filter_window(labels)
        #inputs = torch.randn((1,69,721,1440), device=self.device).detach()
        #labels = torch.randn((1,69,721,1440), device=self.device).detach()

        #print("inputs.shape", inputs.shape, flush=True)
        #print("labels.shape", inputs.shape, flush=True)
        
        if self.diffusion:
            inputs, labels = self.wp_dataset_load_diffusion(self.data_dtype)
        elif self.rit:
            inputs, labels, interval = self.wp_dataset_load(self.data_dtype)
            #assert interval.shape == torch.zeros((1,1)).shape, (interval.shape, inputs.shape, labels.shape)
            #assert interval.dtype == torch.float32
        elif self.wp_load:
            inputs, labels = self.wp_dataset_load(self.data_dtype)
        else:
            inputs, labels = next(self.dataloader)
            inputs = self.slice_sp_window(inputs)
            labels = self.slice_sp_window(labels)

    

        ape = None

        if self.is_first_stage() and not hasattr(self.model, 'ape_generated'):
            #ape_shape = list((1,69,720,1440))
            res = self.dataset.img_resolution
            if not self.wp_load:
                assert res[0] == 721, "other resolutions not implemented"
            #ape_shape = [1, self.dataset.n_channels, res[0], res[1]]
            #assert self.model.condition_channels == self.model.in_channels//2, "condition_channels assumed to be equal to in_channels"
            #assert self.dataset.n_channels == 69, "code assumes 69 channels"
            if self.diffusion:
                ape_shape = (1, self.model_in_channels, 721, 1440)
                ape_sample = np.zeros((1, self.model_in_channels, 721, 1440))
            else:
                ape_shape = (1, self.model_in_channels, 721, 1440)
                ape_sample = np.zeros((1, self.model_in_channels, 721, 1440))
            ape_fn = self.model.ape
            ape = ape_fn(torch.zeros(ape_shape, device=self.device, dtype=self.data_dtype))
            self.model.ape_generated = self.filter_window(ape)

        if self.is_last_stage() and not hasattr(self.model, 'w_lat'):
            #ape_shape = list((1,69,720,1440))
            res = self.dataset.img_resolution
            if not self.wp_load:
                assert res[0] == 721, "other resolutions not implemented"
            #ape_shape = [1, self.dataset.n_channels, res[0], res[1]]
            w_lat_shape = (1, 1, 721, 1440)
            w_lat = torch.cos(torch.deg2rad(torch.linspace(-90, 90, 721))) + self.w_lat_add
            w_lat = w_lat / w_lat.mean()
            w_lat = w_lat.view((1,1,-1,1)).expand(w_lat_shape)
            self.model.w_lat = self.filter_window(w_lat).to(self.device)

        if self.is_first_stage():
            inputs = inputs.to(self.device, dtype=self.data_dtype).detach()
            if self.diffusion:
                pass#handle below after noise stuff
            elif self.rit:
                inputs.requires_grad = True
                interval = interval.to(self.device, dtype=self.data_dtype).detach()
                inputs.interval = True
                self.pipe_buffers["inputs"][buffer_id] = inputs, interval
            else:
                self.pipe_buffers["inputs"][buffer_id] = inputs
                self.pipe_buffers["inputs"][buffer_id].requires_grad = True
            end = time.time()

            #print(self.rank, "ape.shape", self.model.ape_generated.shape, flush=True)
            #print(self.rank, "mask_ape.shape", mask_ape.shape, flush=True)
            #print(self.rank, "inputs.shape", inputs.shape, flush=True)
        
        if self.is_first_stage() or self.is_last_stage():
            labels = labels.to(self.device, dtype=self.data_dtype)
            self.pipe_buffers["labels"][buffer_id] = labels
            #print("labels.shape", labels.shape)
            if self.diffusion:
                #b*w n c
                x = labels
                
                t, z = parallel_TrigFlow_calculate_noise(x, rng=self.rng, nl_rng=self.nl_rng, sigma_min=self.sigma_min, sigma_max=self.sigma_max, sigma_data=self.sigma_data, dtype=self.data_dtype)

                self.pipe_buffers["noise"][buffer_id] = z
                self.pipe_buffers["noise_level"][buffer_id] = t
                condition = inputs
                if self.is_first_stage():
                    cos_t, sin_t = torch.cos(t), torch.sin(t)
                    x_t = cos_t * x + sin_t * z

                    inputs = torch.cat([x_t / self.sigma_data, condition], dim=-1)
                    #assert inputs.size(-1) == 138, inputs.shape
                    self.pipe_buffers["inputs"][buffer_id] = inputs
                    self.pipe_buffers["inputs"][buffer_id].requires_grad = True

        
        #start = time.time()

        #To avoid hiding uneven dataloading. Without this barrier the delays will show up in the next sync
        #In this case that would be the next fwd pass, to be exact the SP all-to-all
        torch.distributed.barrier(self.sp_group)
    def reduce_loss(self):
        if self.is_last_stage():
            loss_tensor = torch.tensor(self.loss/self.GAS)
            torch.distributed.all_reduce(loss_tensor, group=self.weight_group, op=torch.distributed.ReduceOp.SUM)
            if self.rank == self.grid[-1, -1, -1, -1, -1]:
                print(self.rank, "loss s=pp", loss_tensor/self.weight_group_size, flush=True)
                self.tot_loss = (loss_tensor/self.weight_group_size).item()
            self.loss = 0.0
    
    def reduce_grads(self):
        if self.dist_optim:
            self.reduce_acc_grads_distopt()
        else:
            self.reduce_fp32_acc_grads()

    def reduce_fp32_acc_grads(self):
        if self.overlap_p2p_comms:
            wait_handle(self.send_handle_x)
            wait_handle(self.send_handle_dt)
        torch.xpu.synchronize()
        self.reduce_loss()

        torch.distributed.all_reduce(self.grad_bufs, group=self.weight_group, op=torch.distributed.ReduceOp.SUM)
        #After reduce scale.
        self.grad_bufs.mul_(1 / (self.DP*self.GAS))

        if self.grad_clip:
            norm_type = self.grad_clip_norm_type
            grad_norm = torch.norm(self.grad_bufs,p=norm_type)
            total_norm = grad_norm ** norm_type
            self.stage_norm=total_norm.item()
            torch.distributed.all_reduce(total_norm,op=torch.distributed.ReduceOp.SUM,group=self.pp_group)
            total_norm = total_norm.item() ** (1.0 / norm_type)
            self.total_norm=total_norm
            max_norm = self.grad_clip_max_norm
            clip_coeff = max_norm / (total_norm + 1.0e-6)
            if clip_coeff < 1.0:
                self.grad_bufs.detach().mul_(clip_coeff)


        for (param_name_main, param_main),(param_name, param) in zip(self.main_model.named_parameters(),self.model.named_parameters()):
            if not param.requires_grad:
                continue
            assert param.main_grad != None
            param_main.grad = param.main_grad.clone().detach().to(param_main.device)
            torch.nan_to_num(
                param_main.grad,
                nan=0,
                posinf=1e5,
                neginf=-1e5,
                out=param.grad,#TODO check if this ever did anything. Potentially a bug. Why is this set to param.grad instead of param_main.grad?
            )
        
        self.grad_bufs.zero_()
        torch.xpu.synchronize()
    
    def reduce_acc_grads_distopt(self):
        if self.overlap_p2p_comms:
            wait_handle(self.send_handle_x)
            wait_handle(self.send_handle_dt)
        torch.xpu.synchronize()

        self.reduce_loss()
        rs_grads = self.rs_grads
        if rs_grads:
            #reduced_grads = torch.zeros(self.local_num_grad_padded, dtype=self.grad_dtype,device=self.device,requires_grad=False)
            reduced_grads = self.reduce_buf
            torch.distributed.reduce_scatter_tensor(reduced_grads, self.grad_and_param_buf, group=self.weight_group, op=torch.distributed.ReduceOp.SUM)
            #torch.distributed.reduce_scatter(reduced_grads, list(torch.chunk(self.grad_and_param_buf, self.weight_group_size)), group=self.weight_group, op=torch.distributed.ReduceOp.SUM)
            #torch.distributed.reduce_scatter_tensor(reduced_grads, self.grad_and_param_buf.clone(), group=self.weight_group, op=torch.distributed.ReduceOp.SUM)

        else:
            torch.distributed.all_reduce(self.grad_and_param_buf, group=self.weight_group, op=torch.distributed.ReduceOp.SUM)
            opt_rank = self.weight_group_rank
            reduced_grads = self.grad_and_param_buf[opt_rank*self.local_num_grad_padded:(opt_rank+1)*self.local_num_grad_padded].detach()

        #self.local_param_buf.detach().copy_(self.grad_and_param_buf[opt_rank*self.local_num_grad_padded:(opt_rank+1)*self.local_num_grad_padded].detach())
        #After reduce scale.
        reduced_grads.mul_(1 / (self.DP*self.GAS))

        if self.grad_clip:
            norm_type = self.grad_clip_norm_type
            grad_norm = torch.norm(reduced_grads,p=norm_type)
            total_norm = grad_norm ** norm_type
            torch.distributed.all_reduce(total_norm,op=torch.distributed.ReduceOp.SUM,group=self.weight_group)
            self.stage_norm=total_norm.item()
            torch.distributed.all_reduce(total_norm,op=torch.distributed.ReduceOp.SUM,group=self.pp_group)
            total_norm = total_norm.item() ** (1.0 / norm_type)
            self.total_norm=total_norm
            max_norm = self.grad_clip_max_norm
            clip_coeff = max_norm / (total_norm + 1.0e-6)
            if clip_coeff < 1.0:
                reduced_grads.mul_(clip_coeff)

        torch.nan_to_num(
            reduced_grads,
            nan=0,
            posinf=1e5,
            neginf=-1e5,
            out=self.local_param_buf_opt.grad,
        )
        #del reduced_grads
        gc.collect()
        
        #self.grad_and_param_buf.zero_() #No need to zero as we gather into this anyway
        torch.xpu.synchronize()

    def step(self):
        #if self.overlap_p2p_comms:
            #self.comm.Barrier()
            #torch.distributed.barrier(group=self.weight_group)
        #torch.xpu.synchronize()
        if self.base_lr == None:
            self.base_lr = [self.base_lr_num for _ in self.optimizer.param_groups]

        warmup_nimg = self.lr_rampup_img
        lr_rampdown_start = self.cfg.trainer.lr_rampdown_start
        lr_rampdown_length = self.cfg.trainer.lr_rampdown_length
        if self.cur_nimg > lr_rampdown_start + lr_rampdown_length:
            for g, base_lr in zip(self.optimizer.param_groups, self.base_lr):
                min_lr = base_lr * self.lr_min_factor
                g["lr"] = min_lr
        elif self.cur_nimg > lr_rampdown_start:  # linear rampdown
            progress = 1.0 - ((self.cur_nimg-lr_rampdown_start) / lr_rampdown_length)
            for g, base_lr in zip(self.optimizer.param_groups, self.base_lr):
                min_lr = base_lr * self.lr_min_factor
                g["lr"] = min_lr + (base_lr - min_lr) * progress


        elif self.cur_nimg < warmup_nimg:  # linear warmup
            progress = self.cur_nimg / warmup_nimg
            for g, base_lr in zip(self.optimizer.param_groups, self.base_lr):
                min_lr = base_lr * self.lr_min_factor
                g["lr"] = min_lr + (base_lr - min_lr) * progress
        else:  # cosine annealing
            progress_cos = min(
                1.0,
                (self.cur_nimg - warmup_nimg) / (self.total_img - warmup_nimg),
            )
            for g, base_lr in zip(self.optimizer.param_groups, self.base_lr):
                min_lr = base_lr * self.lr_min_factor
                g["lr"] = min_lr + 0.5 * (base_lr - min_lr) * (
                    1 + math.cos(math.pi * progress_cos)
                )

        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.dist_optim:

            #Buffer to update locally. Reduce-scatter grads into this too.
            #self.local_param_buf = torch.zeros(local_num_grad_padded, dtype=torch.float32,device=self.device,requires_grad=False)
            #self.local_param_buf_opt = torch.nn.Parameter(self.local_param_buf[:local_num_grad_unpadded])
            
            #Accumulate grads into this. Allgather also into this.
            #self.grad_and_param_buf = torch.zeros(padded_total_buf_size, dtype=torch.float32,device=self.device,requires_grad=False)
            torch.distributed.all_gather_into_tensor(self.grad_and_param_buf, self.local_param_buf, group=self.weight_group)

            #Copy main model weights to model weights
            total_numel = 0
            for param in self.model.parameters():
                if param.requires_grad:
                    total_numel += param.nelement()
            start_index = total_numel
            for param in self.model.parameters():
                if param.requires_grad:
                    start_index -= param.nelement()
                    shape = param.shape
                    end_index = start_index + shape.numel()
                    assert end_index <= total_numel, 'requested tensor is out of the buffer range.'
                    #Assign main grad
                    buffer_tensor = self.grad_and_param_buf[start_index:end_index]
                    buffer_tensor = buffer_tensor.view(shape)
                    param.detach().copy_(buffer_tensor.detach())
            gc.collect()
        else:
            #Copy main model weights to model weights
            for param, main_param in zip(self.model.parameters(),self.main_model.parameters()):
                if param.requires_grad:
                    param.data.copy_(main_param.data)
                    assert param.data.dtype == self.local_model_dtype
        
        # EMA update
        if self.ema != None and self.cur_nimg < lr_rampdown_start + lr_rampdown_length:
            ema_halflife_nimg = self.ema_halflife_img
            if self.ema_rampup_ratio is not None:
                ema_halflife_nimg = min(
                    ema_halflife_nimg, self.cur_nimg * self.ema_rampup_ratio
                )
            
            if self.cur_nimg > lr_rampdown_start:  # linear rampdown
                base_halflife = self.ema_halflife_img
                progress = 1.0 - ((self.cur_nimg-lr_rampdown_start) / lr_rampdown_length)
                min_halflife = base_halflife * self.cfg.trainer.lr_rampdown_ema_min_ratio
                ema_halflife_nimg = min_halflife + (base_halflife - min_halflife) * progress

            ema_beta = 0.5 ** ((self.GAS * self.DP) / max(ema_halflife_nimg, 1e-8))
            for p_ema, p_net in zip(self.ema.parameters(), self.main_model.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))


        self.cur_nimg += self.GAS * self.DP

    def fwd(self, buffer_id):
        if self.is_first_stage():
            inputs = self.pipe_buffers["inputs"][buffer_id]
            interval = None
            if self.diffusion:
                interval = self.pipe_buffers["noise_level"][buffer_id]
                assert interval.shape == (1,1), interval.shape
            elif self.rit:
                inputs, interval = inputs
                assert interval.shape == (1,1), interval.shape
            
            x = \
                self.model(
                    inputs,
                    None,
                    sigma_data=self.sigma_data,
                    condition=None,
                    auxilary=None,
                    interval=interval
                )
            if self.diffusion or self.rit:
                x, dt = x
                assert dt.shape == self.recv_buf_dt.shape, (dt.shape, self.recv_buf_dt.shape)
                self.pipe_buffers["dt"][buffer_id] = dt
            self.pipe_buffers["outputs"][buffer_id] = x
        else:
            """if self.rank in self.grid[0,0,0,2,:].flatten().tolist():
                start = time.time()
                torch.distributed.barrier(self.sp_group)
                end = time.time()
                print(self.sp_rank, "sp_barrier0", end-start)
                self.pipe_buffers["outputs"][buffer_id] = self.model(self.pipe_buffers["inputs"][buffer_id],self.pipe_buffers["t"][buffer_id],self.sp_group, benchmark=True)
            else:"""

            ws_y, ws_x = self.window_size
            wc_y_tot = 720//ws_y
            wc_x_tot = 1440//ws_x
            wc_y_loc = (wc_y_tot//self.WP_Y)
            wc_x_loc = (wc_x_tot//self.WP_X)
            assert self.pipe_buffers["inputs"][buffer_id].dtype == torch.bfloat16
            inputs = self.pipe_buffers["inputs"][buffer_id]
            dims = inputs.shape
            assert len(dims) == 3, "Not implemented for more dims"
            inputs = inputs.reshape((wc_y_loc,wc_x_loc,dims[1],dims[2]))
            if self.pp_rank%2 == 0:#previous stage shifts up left
                if self.wp_y_rank == self.WP_Y-1 and self.sp_rank >5:#bottom half
                    inputs = inputs.roll(-1,0)
                if self.wp_x_rank == self.WP_X-1 and ((self.sp_rank > 2 and self.sp_rank < 6) or self.sp_rank > 8):#right half
                    inputs = inputs.roll(-1,1)
            inputs = inputs.reshape(dims)

            if self.rit or self.diffusion:
                dt = self.pipe_buffers["dt"][buffer_id]
            else:
                dt = None
            outputs = self.model(inputs,None,dt,self.sp_group)

            dims = outputs.shape
            assert len(dims) == 3, "Not implemented for more dims"
            outputs = outputs.reshape((wc_y_loc,wc_x_loc,dims[1],dims[2]))

            if self.pp_rank%2 == 0:#previous stage shifts up left
                if self.wp_y_rank == self.WP_Y-1 and self.sp_rank >5:#bottom half
                    outputs = outputs.roll(1,0)
                if self.wp_x_rank == self.WP_X-1 and ((self.sp_rank > 2 and self.sp_rank < 6) or self.sp_rank > 8):#right half
                    outputs = outputs.roll(1,1)
            outputs = outputs.reshape(dims)
            
            self.pipe_buffers["outputs"][buffer_id] = outputs

    def bwd(self, buffer_id):
        output = self.pipe_buffers["outputs"][buffer_id]
        if self.is_last_stage():
            target = self.pipe_buffers["labels"][buffer_id]

            if self.cfg.model.enhanced_channels4:
                #Channel weights. Hardcoded, of course
                w_var = torch.tensor([0.1562, 0.0156, 0.0156, 0.0156, 0.0013, 0.0026, 0.0039, 0.0052, 0.0065,
                        0.0078, 0.0104, 0.0130, 0.0156, 0.0182, 0.0220, 0.0240, 0.0259, 0.0013,
                        0.0026, 0.0039, 0.0052, 0.0065, 0.0078, 0.0104, 0.0130, 0.0156, 0.0182,
                        0.0220, 0.0240, 0.0259, 0.0013, 0.0026, 0.0039, 0.0052, 0.0065, 0.0078,
                        0.0104, 0.0130, 0.0156, 0.0182, 0.0220, 0.0240, 0.0259, 0.0013, 0.0026,
                        0.0039, 0.0052, 0.0065, 0.0078, 0.0104, 0.0130, 0.0156, 0.0182, 0.0220,
                        0.0240, 0.0259, 0.0013, 0.0026, 0.0039, 0.0052, 0.0065, 0.0078, 0.0104,
                        0.0130, 0.0156, 0.0182, 0.0220, 0.0240, 0.0259, 0.0156], dtype=torch.float32, device=output.device).view((1,1,-1))#[288, 75, 70] b*wc s c
            else:
                #Channel weights. Hardcoded, of course
                w_var = torch.tensor([0.15873015, 0.01587301, 0.01587301, 0.01587301, 0.00131726,
                    0.00263453, 0.00395179, 0.00526905, 0.00658631, 0.00790358,
                    0.0105381 , 0.01317263, 0.01580715, 0.01844168, 0.02239347,
                    0.02436936, 0.02634525, 0.00131726, 0.00263453, 0.00395179,
                    0.00526905, 0.00658631, 0.00790358, 0.0105381 , 0.01317263,
                    0.01580715, 0.01844168, 0.02239347, 0.02436936, 0.02634525,
                    0.00131726, 0.00263453, 0.00395179, 0.00526905, 0.00658631,
                    0.00790358, 0.0105381 , 0.01317263, 0.01580715, 0.01844168,
                    0.02239347, 0.02436936, 0.02634525, 0.00131726, 0.00263453,
                    0.00395179, 0.00526905, 0.00658631, 0.00790358, 0.0105381 ,
                    0.01317263, 0.01580715, 0.01844168, 0.02239347, 0.02436936,
                    0.02634525, 0.00131726, 0.00263453, 0.00395179, 0.00526905,
                    0.00658631, 0.00790358, 0.0105381 , 0.01317263, 0.01580715,
                    0.01844168, 0.02239347, 0.02436936, 0.02634525], dtype=torch.float32, device=output.device).view((1,1,-1))#[288, 75, 69] b*wc s c
            if self.diffusion:
                x = target[...,:self.model_out_channels]
                t = self.pipe_buffers["noise_level"][buffer_id]
                z = self.pipe_buffers["noise"][buffer_id][...,:self.model_out_channels]
                
                cos_t, sin_t = torch.cos(t), torch.sin(t)
                v_t = cos_t * z - sin_t * x
                loss = ((self.model.w_lat * w_var * (self.sigma_data * output - v_t) ** 2)).sum(dim=2).mean()
            else:
                loss = ((self.model.w_lat * w_var * (output - target) ** 2)).sum(dim=2).mean()
            self.loss += loss.item()
            if not self.benchmark and math.isnan(loss):
                print("nan loss on rank", self.rank, flush=True)
                raise NotImplementedError
            loss.backward()
            return
        #assert self.grad_buf_id == buffer_id, "something wrong, grads not aligned to the output buffer"
        assert output!=None, "output is None"
        assert self.pipe_buffers["grad_x"][buffer_id] != None, f"grad is None, {self.pp_rank}, {buffer_id}"
        if self.is_first_stage():
            torch.autograd.backward(tensors=output.to(self.local_model_dtype), grad_tensors=self.pipe_buffers["grad_x"][buffer_id].to(self.local_model_dtype))
            if self.rit or (self.diffusion and not self.layerwise_t_emb):
                dt = self.pipe_buffers["dt"][buffer_id]
                torch.autograd.backward(tensors=dt, grad_tensors=self.pipe_buffers["grad_t"][buffer_id])
            #torch.autograd.backward(tensors=[output], grad_tensors=[self.pipe_buffers['t'][buffer_id].grad])
            
        else:
            torch.autograd.backward(tensors=output, grad_tensors=self.pipe_buffers["grad_x"][buffer_id])

    def p2p_send(self, data, rank, group, tag=0):
        if not self.benchmark:
            assert not torch.isnan(data).any(), "trying to send NaN"
        if self.overlap_p2p_comms:
            send_handle = self.comm.Isend(data.detach(), dest=rank, tag=tag)
            return send_handle
        else:
            torch.distributed.send(data.detach(), rank, group=group)
            return None
        
    def p2p_recv(self, buffer, rank, group, tag=0):
        if self.overlap_p2p_comms:
            recv_handle = self.comm.Irecv(buffer, source=rank, tag=tag)
            return recv_handle
        else:
            torch.distributed.recv(buffer, rank, group=group)
            return None
    
    def send(self, buffer_id):
        outputs = self.pipe_buffers['outputs'][buffer_id]
        if self.rit or self.diffusion:
            dt = self.pipe_buffers["dt"][buffer_id]
        
        if self.overlap_p2p_comms:
            outputs = outputs.to("cpu")
            dt = dt.to("cpu")
        
        wait_handle(self.send_handle_x)
        wait_handle(self.send_handle_dt)
        self.send_handle_x = self.p2p_send(outputs, self.next_rank, group=self.pp_comm_group_next, tag=1)
        if self.rit or self.diffusion:
            self.send_handle_dt = self.p2p_send(dt, self.next_rank, group=self.pp_comm_group_next, tag=1)

    def send_grad(self, buffer_id):
        grad_x = self.pipe_buffers['inputs'][buffer_id].grad
        if self.rit or (self.diffusion and not self.layerwise_t_emb):
            grad_t = self.pipe_buffers["dt"][buffer_id].grad

        if self.overlap_p2p_comms:
            assert grad_x != None, f"grad_x is none {self.pp_rank} {buffer_id}"
            grad_x = grad_x.to("cpu") 
            grad_t = grad_t.to("cpu")
        wait_handle(self.send_handle_x)
        wait_handle(self.send_handle_dt)
        self.send_handle_x = self.p2p_send(grad_x, self.prev_rank, group=self.pp_comm_group_prev, tag=2)
        if self.rit or (self.diffusion and not self.layerwise_t_emb):
            
            grad_t = grad_t.to(torch.float32) + self.grad_recv_t
            assert grad_t.shape == self.grad_recv_t.shape, (grad_t.shape, self.grad_recv_t.shape)
            self.send_handle_dt = self.p2p_send(grad_t, self.prev_rank, group=self.pp_comm_group_prev, tag=2)

    def recv_postprocess(self, data, buffer, buffer_id):
        if self.overlap_p2p_comms:
            self.pipe_buffers[buffer][buffer_id] = data.detach().clone().to(self.device)
        else:
            self.pipe_buffers[buffer][buffer_id] = data.detach().clone()
        self.pipe_buffers[buffer][buffer_id].requires_grad = True

    def submit_recv(self, buffer_id):
        self.recv_handle_x = self.p2p_recv(self.recv_buf_x, self.prev_rank, group=self.pp_comm_group_prev, tag=1)
        if self.rit or self.diffusion:
            self.recv_handle_dt = self.p2p_recv(self.recv_buf_dt, self.prev_rank, group=self.pp_comm_group_prev, tag=1)

    def process_recv(self, buffer_id):
        #Do this after to post both receives first in case of RIT
        wait_handle(self.recv_handle_x)
        self.recv_postprocess(self.recv_buf_x, "inputs", buffer_id)
        if self.rit or self.diffusion:
            wait_handle(self.recv_handle_dt)
            self.recv_postprocess(self.recv_buf_dt, "dt", buffer_id)
    
    def recv(self, buffer_id):
        self.submit_recv(buffer_id)
        self.process_recv(buffer_id)

    def submit_recv_grad(self, buffer_id):
        self.recv_handle_grad_x = self.p2p_recv(self.grad_recv_x, self.next_rank, group=self.pp_comm_group_next, tag=2)
        if self.rit or (self.diffusion and not self.layerwise_t_emb):
            self.recv_handle_grad_dt = self.p2p_recv(self.grad_recv_t, self.next_rank, group=self.pp_comm_group_next, tag=2)

    def process_grad(self, buffer_id):
        wait_handle(self.recv_handle_grad_x)
        if self.overlap_p2p_comms:
            self.pipe_buffers["grad_x"][buffer_id] = self.grad_recv_x.clone().to(self.device)
            assert self.pipe_buffers["grad_x"][buffer_id] != None, "grad is None"
        else:
            self.pipe_buffers["grad_x"][buffer_id] = self.grad_recv_x

        wait_handle(self.recv_handle_grad_dt)
        if self.overlap_p2p_comms:
            self.pipe_buffers["grad_t"][buffer_id] = self.grad_recv_t.clone().to(self.device)
        else:
            self.pipe_buffers["grad_t"][buffer_id] = self.grad_recv_t
        
        
    def recv_grad(self, buffer_id):
        self.submit_recv_grad(buffer_id)
        self.process_grad(buffer_id)


    def save_checkpoint(self):
        if self.dist_optim:
            #TODO saving a checkpoint with dist optimizer
            return
        it = self.cur_nimg // (self.GAS*self.DP)
        if it%1000 == 0 and it>0:
            fname = os.path.join(self.checkpoint_path, f"it_{it}_checkpoint_PP{self.pp_rank}.pth")
        else:
            fname = os.path.join(self.checkpoint_path, f"checkpoint_PP{self.pp_rank}.pth")
        checkpoint = {
            'cur_nimg': self.cur_nimg,
            'model_state_dict': self.model.state_dict(),
            'main_model_state_dict': self.main_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Add any other relevant information
        }
        if self.ema != None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        torch.save(checkpoint, fname)
        return
    
    def load_checkpoint(self):
        #TODO
        fname = os.path.join(self.checkpoint_path, f"checkpoint_PP{self.pp_rank}.pth")
        try:
            checkpoint = torch.load(fname,map_location="cpu",weights_only=True)
        except:
            if self.rank == 0:
                print("Could not find a checkpoint from", self.checkpoint_path, flush=True)
            return
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.main_model.load_state_dict(checkpoint['main_model_state_dict'])
        if "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cur_nimg = checkpoint['cur_nimg']
        self.setup_noise_rng(seed=self.cur_nimg+self.noise_rng_seed+self.wp_x_rank*self.WP_Y*self.DP*self.SP+self.wp_y_rank*self.DP*self.SP+self.dp_rank*self.SP+self.sp_rank)
        self.setup_noise_level_rng(seed=self.cur_nimg+self.noise_rng_seed+self.dp_rank)
        torch.distributed.barrier()
        if self.rank == 0:
            print("loaded a checkpoint from", self.checkpoint_path, flush=True)
        return

    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: step,
        schedule.ReduceGrads: reduce_grads,
        schedule.ReduceTiedGrads: placeholder,
        schedule.LoadMicroBatch: load,
        schedule.ForwardPass: fwd,
        schedule.BackwardPass: bwd,
        schedule.SendActivation: send,
        schedule.RecvActivation: recv,
        schedule.SendGrad: send_grad,
        schedule.RecvGrad: recv_grad,
        SubmitRecvGrad: submit_recv_grad,
        SubmitRecvActivation: submit_recv,
        ProcessActivation: process_recv,
        ProcessGrad: process_grad,
    }

    _STR_INSTR_MAP = {
        schedule.OptimizerStep: "step",
        schedule.ReduceGrads: "reduce_grads",
        schedule.ReduceTiedGrads: "nothing",
        schedule.LoadMicroBatch: "load_data",
        schedule.ForwardPass: "fwd",
        schedule.BackwardPass: "bwd",
        schedule.SendActivation: "send_act",
        schedule.RecvActivation: "recv_act",
        schedule.SendGrad: "send_grad",
        schedule.RecvGrad: "recv_grad",
        SubmitRecvGrad: "submit_recv_grad",
        SubmitRecvActivation: "submit_recv",
        ProcessActivation: "process_recv",
        ProcessGrad: "process_grad",
    }

    def setup_schedule(self):
        if self.overlap_p2p_comms:
            self.schedule = LessCustomTrainSchedule(micro_batches=self.GAS, stages=self.PP, stage_id=self.pp_rank)
        else:
            self.schedule = schedule.TrainSchedule(micro_batches=self.GAS, stages=self.PP, stage_id=self.pp_rank)
    
    def exec_schedule(self):
        timers = {}
        torch.xpu.synchronize()
        for step_cmds in self.schedule:
            # For each instruction in the step
            for cmd in step_cmds:
                instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                start = time.time()
                instr(**cmd.kwargs)
                end = time.time()
                if self._STR_INSTR_MAP[type(cmd)] in timers:
                    timers[self._STR_INSTR_MAP[type(cmd)]] += end-start
                else:
                    timers[self._STR_INSTR_MAP[type(cmd)]] = end-start
        torch.xpu.synchronize()
        report_timings = self.cur_nimg == (self.GAS*self.DP*4) or self.cur_nimg == (self.GAS*self.DP*5)
        #report_timings = False
        if report_timings and self.rank in self.grid[0,0,0,:,0].flatten().tolist():
            print(self.pp_rank, "times:")
            for key in timers:
                print(self.pp_rank, key, timers[key])
            print("",flush=True)

        if self.cur_nimg % (self.GAS*self.DP*100) == 0 and not self.benchmark:
            start = time.time()
            #Cant have these barriers in save_checkpoint as not everyone participates and there is no comm group for this.
            torch.distributed.barrier()
            if self.rank in self.grid[0,0,0,:,0].flatten().tolist():
                self.save_checkpoint()
            torch.distributed.barrier()
            end = time.time()
            if self.rank==0:
                print("saved a checkpoint to", self.checkpoint_path, "in", end-start, flush=True)

        gc.collect()
        return timers

    def setup_noise_rng(self, seed=0):
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)
    def setup_noise_level_rng(self, seed=0):
        self.nl_rng = torch.Generator(device=self.device)
        self.nl_rng.manual_seed(seed)

    def setup_buffers(self, dim, seq, batch_size):
        num_buffers = self.schedule.num_pipe_buffers()
        self.pipe_buffers = {
            'inputs': [],  # batch input and received activations
            'labels': [],  # labels from batch input
            'outputs': [],  # activations
            'sigma': [],  # activations
            'noise': [],  # activations
            'noise_level': [],  # activations
            'dt': [],  # activations
            'grad_x': [],  # grads
            'grad_t': [],  # grads
        }
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_buffers)
        self.grads_recvd = None

        if self.overlap_p2p_comms:
            self.comm_buf_device = "cpu" 
        else:
            self.comm_buf_device = self.device

        self.recv_buf_x = torch.zeros((batch_size, (seq//self.SP), dim), device=self.comm_buf_device, dtype=torch.bfloat16)
        self.recv_buf_dt = torch.zeros((1, dim), device=self.comm_buf_device, dtype=torch.bfloat16)
        self.grad_recv_x = torch.zeros((batch_size, (seq//self.SP), dim), device=self.comm_buf_device, dtype=torch.bfloat16)
        self.grad_recv_t = torch.zeros((1, dim), device=self.comm_buf_device, dtype=torch.float32)


        self.send_handle_x = None
        self.send_handle_dt = None
        self.recv_handle_x = None
        self.recv_handle_dt = None
        self.recv_handle_grad_x = None
        self.recv_handle_grad_dt = None

    def setup_grad_acc_locopt(self):
        total_numel = 0
        for param in self.model.parameters():
            if param.requires_grad:
                total_numel += param.data.nelement()
        self.grad_bufs = torch.zeros(total_numel, dtype=torch.float32,device=self.device,requires_grad=False)
        start_index = total_numel
        if self.rank == self.grid[0,0,0,self.pp_rank,0]:
            print("Number of parameters PP={} : {:.2f}B".format(self.pp_rank,total_numel/1e9), flush=True)

        def _make_param_hook(param):
            def param_hook(*unused):
                # Add the gradient to the buffer.
                if param.grad is not None:
                    param.main_grad.add_(param.grad.data)
                    # deallocate grad memory.
                    param.grad = None
            return param_hook
        
        #Store to maintain access
        self.grad_accs = []

        for param in self.model.parameters():
            if param.requires_grad:
                start_index -= param.data.nelement()
                shape = param.data.shape
                end_index = start_index + shape.numel()
                assert end_index <= total_numel, 'requested tensor is out of the buffer range.'
                #Assign main grad
                buffer_tensor = self.grad_bufs[start_index:end_index]
                buffer_tensor = buffer_tensor.view(shape)
                param.main_grad = buffer_tensor

                # Expand to get access to grad_fn.
                param_tmp = param.expand_as(param)
                #Register gradient accumulation hook:
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(_make_param_hook(param))
                self.grad_accs.append(grad_acc)

    def setup_grad_acc_distopt(self):
        total_numel = 0
        for param in self.model.parameters():
            if param.requires_grad:
                total_numel += param.data.nelement()
        opt_group_size = self.weight_group_size
        self.local_num_grad_padded = math.ceil(total_numel/opt_group_size)
        padded_total_buf_size = self.local_num_grad_padded*opt_group_size
        
        opt_rank = self.weight_group_rank
        last_rank = opt_rank == self.weight_group_size-1
        last_size = (total_numel-(opt_group_size-1)*self.local_num_grad_padded)
        if last_rank:
            self.local_num_grad_unpadded=last_size
        else:
            self.local_num_grad_unpadded=self.local_num_grad_padded

        assert last_size > 0, "something off with grad bufs"

        #Buffer to update locally. Reduce-scatter grads into this too.
        self.local_param_buf = torch.zeros(self.local_num_grad_padded, dtype=self.grad_dtype,device=self.device,requires_grad=False)
        self.reduce_buf = torch.zeros(self.local_num_grad_padded, dtype=self.grad_dtype,device=self.device,requires_grad=False)
        self.local_param_buf_opt = torch.nn.Parameter(self.local_param_buf[:self.local_num_grad_unpadded])
        
        #Accumulate grads into this. Allgather also into this.
        self.grad_and_param_buf = torch.zeros(padded_total_buf_size, dtype=self.grad_dtype,device=self.device,requires_grad=False)


        
        #Copy model weights to main model weights 
        total_numel = 0
        for param in self.model.parameters():
            if param.requires_grad:
                total_numel += param.nelement()
        start_index = total_numel
        for param in self.model.parameters():
            if param.requires_grad:
                start_index -= param.detach().nelement()
                shape = param.shape
                end_index = start_index + shape.numel()
                assert end_index <= total_numel, 'requested tensor is out of the buffer range.'
                self.grad_and_param_buf[start_index:end_index].detach().copy_(param.detach().view(-1))
        self.local_param_buf.detach().copy_(self.grad_and_param_buf[opt_rank*self.local_num_grad_padded:(opt_rank+1)*self.local_num_grad_padded].detach())

        if self.rank == self.grid[0,0,0,self.pp_rank,0]:
            print("Number of parameters PP={} : {:.2f}B".format(self.pp_rank,total_numel/1e9), flush=True)

        def _make_param_hook(param):
            def param_hook(*unused):
                # Add the gradient to the buffer.
                if param.grad is not None:
                    param.main_grad.add_(param.grad.detach())
                    # deallocate grad memory.
                    param.grad = None
            return param_hook
        
        #Store to maintain access
        self.grad_accs = []

        start_index = total_numel
        for param in self.model.parameters():
            if param.requires_grad:
                start_index -= param.nelement()
                shape = param.shape
                end_index = start_index + shape.numel()
                assert end_index <= total_numel, 'requested tensor is out of the buffer range.'
                #Assign main grad
                buffer_tensor = self.grad_and_param_buf[start_index:end_index]
                buffer_tensor = buffer_tensor.view(shape)
                param.main_grad = buffer_tensor

                # Expand to get access to grad_fn.
                param_tmp = param.expand_as(param)
                #Register gradient accumulation hook:
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(_make_param_hook(param))
                self.grad_accs.append(grad_acc)

    def init_params(self, model, dim, cfg):
        #Normal init
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.RMSNorm):
                torch.nn.init.normal_(m.weight, mean=0.0, std=(2 / (5 * dim)) ** 0.5)

        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.RMSNorm):
                if cfg.model.init_scale_test15:
                    layers = (self.PP-2)*cfg.model.sublayers
                    scale = (1/(2*layers))**0.5
                    if hasattr(m, "is_time_emb"):
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name,  "to", cfg.model.time_emb_init)
                        nn.init.trunc_normal_(m.weight, std=cfg.model.time_emb_init)
                    elif "head" in name:
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name, "to", 0)
                        nn.init.zeros_(m.weight)
                    elif hasattr(m, "is_emb"):
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name, "to", cfg.model.token_emb_init_std)
                        nn.init.trunc_normal_(m.weight, std=cfg.model.token_emb_init_std)
                    elif ("wo" in name or "w2" in name):
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name, "to", 0)
                        nn.init.zeros_(m.weight)
                    elif "modulation.0" in name:
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name, "to", 0)
                        if cfg.model.modulation_0_init == 0:
                            nn.init.zeros_(m.weight)
                        else:
                            nn.init.trunc_normal_(m.weight, std=cfg.model.modulation_0_init)
                        #nn.init.zeros_(m.weight)
                    elif "modulation.2" in name:
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name, "to", cfg.model.modulation_2_init)
                        nn.init.zeros_(m.weight)
                        #nn.init.zeros_(m.weight)
                    else:
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name, "to", ((2 / (5 * dim)) ** 0.5))
                        nn.init.trunc_normal_(m.weight, std=(2 / (5 * dim)) ** 0.5)
                elif cfg.model.init_scale_test14:
                    layers = (self.PP-2)*cfg.model.sublayers
                    scale = (1/(2*layers))**0.5
                    if hasattr(m, "is_time_emb"):
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name,  "to", cfg.model.time_emb_init)
                        nn.init.trunc_normal_(m.weight, std=cfg.model.time_emb_init)
                    elif "head" in name:
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name, "to", cfg.model.head_init_std)
                        nn.init.trunc_normal_(m.weight, std=cfg.model.head_init_std)
                    elif hasattr(m, "is_emb"):
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name, "to", cfg.model.token_emb_init_std)
                        nn.init.trunc_normal_(m.weight, std=cfg.model.token_emb_init_std)
                    elif ("wo" in name or "w2" in name):
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name, "to", ((2 / (5 * dim)) ** 0.5)*scale)
                        nn.init.trunc_normal_(m.weight, std=((2 / (5 * dim)) ** 0.5)*scale)
                    elif "modulation.0" in name:
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name, "to", cfg.model.modulation_0_init)
                        if cfg.model.modulation_0_init == 0:
                            nn.init.zeros_(m.weight)
                        else:
                            nn.init.trunc_normal_(m.weight, std=cfg.model.modulation_0_init)
                        #nn.init.zeros_(m.weight)
                    elif "modulation.2" in name:
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name, "to", cfg.model.modulation_2_init)
                        if cfg.model.modulation_2_init == 0:
                            nn.init.zeros_(m.weight)
                        else:
                            nn.init.trunc_normal_(m.weight, std=cfg.model.modulation_2_init)
                        #nn.init.zeros_(m.weight)
                    else:
                        if self.wp_x_rank==0 and self.wp_y_rank==0 and self.sp_rank==0 and self.dp_rank==0:
                            print("initializing weights of", m, name, "to", ((2 / (5 * dim)) ** 0.5))
                        nn.init.trunc_normal_(m.weight, std=(2 / (5 * dim)) ** 0.5)
                elif hasattr(m, "is_emb") or "head" in name or "modulation" in name:
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.trunc_normal_(m.weight, std=(2 / (5 * dim)) ** 0.5)

        #Sync with all ranks
        for param in model.parameters():
            torch.distributed.broadcast(param.data,
                                        src=self.grid[0,0,0,self.pp_rank,0],
                                        group=self.weight_group)
    def warmup(self, seq, batch_size, dim):
        if not self.is_first_stage() and not self.is_last_stage():
            in_x = torch.randn((batch_size, (seq//self.SP), dim), device=self.device, dtype=self.local_model_dtype)
            in_t = torch.zeros((1, dim), device=self.device, dtype=self.local_model_dtype)
            in_dt = torch.zeros((1, dim), device=self.device, dtype=self.local_model_dtype)
            target = torch.randn((batch_size, (seq//self.SP), dim), device=self.device, dtype=self.local_model_dtype)

            for i in range(3):
                out = self.model(in_x,in_t,in_dt,self.sp_group)
                loss_fn = nn.MSELoss()
                loss = loss_fn(out,target)
                loss.backward()
            benchmark_startup = True
            if benchmark_startup:
                torch.distributed.barrier(group=self.sp_group)
                torch.xpu.synchronize()
                start = time.time()
                for i in range(3):
                    out = self.model(in_x,in_t,in_dt,self.sp_group)
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(out,target)
                    loss.backward()
                torch.xpu.synchronize()
                end = time.time()
            if not self.dist_optim:
                self.grad_bufs.zero_()
            else:
                self.grad_and_param_buf.zero_()
            if hasattr(self, "optimizer"):
                self.optimizer.zero_grad()

            if benchmark_startup:
                time_taken=end-start
                avg_time = torch.tensor(time_taken)
                time.sleep(1.0)
                torch.distributed.barrier(group=self.weight_group)
                time.sleep(0.1)
                torch.distributed.all_reduce(avg_time,group=self.weight_group)
                if self.weight_group_rank == 0:
                    print("weight_group", self.pp_rank, "finished", flush=True)
                if time_taken > (avg_time.item()/self.weight_group_size)*1.08:
                    print("ALERT, slow warmup on rank {}, ({:.1f}%), time taken: {:.3f}, avg: {:.3f}".format(self.rank, 100*(time_taken/(avg_time/self.weight_group_size)), time_taken, avg_time/self.weight_group_size), flush=True)


