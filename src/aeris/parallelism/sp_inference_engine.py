
import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import numpy as np
from einops import rearrange

from deepspeed.runtime.pipe import schedule
from deepspeed.runtime.pipe.schedule import LoadMicroBatch, SendActivation, RecvActivation, ForwardPass, BufferOpInstruction

from types import MethodType
import copy
import math
import time
import os
import gc
import shutil
from glob import glob

####
#This file is practically copy-paste and search-replace of the pipeline_engine.py. Sorry... It would be possible to do inference with just that one instead with some changes.
####

def _is_even(x):
    return x % 2 == 0


class SendResult(BufferOpInstruction):
    """Returns the result
    """
    pass

class RecvResult(BufferOpInstruction):
    """Returns the result
    """
    pass

class ReturnResult(BufferOpInstruction):
    """Returns the result
    """
    pass


class EnsembleInferenceSchedule(schedule.PipeSchedule):
    """A schedule for inferencing batches using pipeline parallelism.
    Inspired by DeepSpeed schedules. 
    """

    def steps(self):
        """"""
        #assert self.micro_batches == self.stages
        total_steps = self.micro_batches + self.stages - 1
        for step_id in range(total_steps):
            cmds = []
            micro_batch_id = step_id - self.stage_id
            prev_micro_batch_id = micro_batch_id-1

            curr_buffer = micro_batch_id
            prev_buffer = prev_micro_batch_id

            if self.is_first_stage:
                if self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            if _is_even(self.stage_id):
                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(prev_buffer):
                        cmds.append(SendActivation(prev_buffer))
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(curr_buffer):
                        cmds.append(RecvActivation(curr_buffer))
            else:
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(curr_buffer):
                        cmds.append(RecvActivation(curr_buffer))

                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(prev_buffer):
                        cmds.append(SendActivation(prev_buffer))

            if self._valid_micro_batch(curr_buffer):
                cmds.append(ForwardPass(curr_buffer))

            if self.is_last_stage:
                #Send back to beginning:
                if self._valid_micro_batch(curr_buffer):
                    cmds.append(SendResult(curr_buffer))
            if self.is_first_stage:
                #Something to receive:
                if self._valid_micro_batch(micro_batch_id - self.stages + 1):
                    cmds.append(ReturnResult(micro_batch_id - self.stages + 1))
            

            yield cmds

    def num_pipe_buffers(self):
        return self.micro_batches


class InferenceEngine():
    def __init__(
            self,
            cfg,
            PP,
            SP,
            WP_X,
            WP_Y,
            n_members,
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
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.PP = PP
        self.SP = SP
        assert SP== 12, "No ParallelEngine implementation for SP!=12"
        self.WP_X = WP_X
        self.WP_Y = WP_Y
        self.n_members = n_members
        self.device = device
        self.window_size = init_model_kwargs["window_size"]
        self.dim = init_model_kwargs["dim"]
        self.ema_halflife_img = ema_halflife_img
        self.ema_rampup_ratio = ema_rampup_ratio
        self.lr_rampup_img = lr_rampup_img
        self.lr_min_factor = lr_min_factor
        self.total_img = total_img
        self.base_lr = None
        self.checkpoint_path = checkpoint_path
        self.sigma_data = sigma_data
        self.sigma_min= cfg.model.sigma_min
        self.sigma_max = cfg.model.sigma_max
        self.rit = init_model_kwargs["rit"]
        self.diffusion = init_model_kwargs["diffusion"]
        self.model_in_channels = init_model_kwargs["model_in_channels"]
        self.model_out_channels = init_model_kwargs["model_out_channels"]


        self.noise_rng_seed = 0

        self.grid_init()
        self.setup_schedule()
        self.setup_buffers(init_model_kwargs["dim"], seq, batch_size)

        self.data_dtype = data_dtype
        self.model_dtype = model_dtype
        self.local_model_dtype = torch.float32 if self.data_dtype==torch.float32 and (self.is_first_stage() or self.is_last_stage()) else torch.bfloat16 

        self.model = init_model_fn(self, cfg, **init_model_kwargs).to(device=self.device, dtype=self.local_model_dtype)
        #self.model = copy.deepcopy(self.main_model)
        self.cur_nimg = 0
        #self.samples_loaded = 0
        
        
    
    def sp_sample_shard(self, inputs, buffer_id):
        ws_y, ws_x = self.window_size
        wc_y_tot = 720//ws_y
        wc_x_tot = 1440//ws_x
        wc_y_loc = (wc_y_tot//self.WP_Y)
        wc_x_loc = (wc_x_tot//self.WP_X)
        wc_local = wc_y_loc*wc_x_loc
        channels = self.model_in_channels if self.diffusion else 69
        input_res = torch.zeros((wc_local, ws_y*ws_x//self.SP, channels), dtype=self.data_dtype, device=self.device)

        if self.sp_rank == 0:
            s = self.SP//4
            #b: batch, c: channels, wc: window count, ws_(h): window size (half), d/e constant two, 
            assert buffer_id < len(inputs), ("buffer_id < len(inputs)", buffer_id, len(inputs))
            inputs = inputs[buffer_id:buffer_id+1].clone()
            #print("inputs.shape", inputs.shape) #torch.Size([1, 69, 360, 720])
            inputs = rearrange(inputs, "b c (wc_y d ws_y_h) (wc_x e ws_x_h) -> (b wc_y wc_x) c (d e) (ws_y_h ws_x_h)", d=2, e=2, ws_y_h=ws_y//2, ws_x_h=ws_x//2)
            inputs = rearrange(inputs, "b c w (s n) -> b (w s) n c", s=s)#w=4,s=SP/4 n is tokens left from window after SP division. normally here we have:(b wc_y wc_x) (ws_y  ws_x) c
            input_list = [item[:,0,:,:].contiguous() for item in inputs.chunk(self.SP, dim=1)]
            #print("input_list[0].shape", input_list[0].shape) #torch.Size([288, 75, 69])
            torch.distributed.scatter(input_res, scatter_list=input_list, src=self.rank,group=self.sp_group)
        else:
            torch.distributed.scatter(input_res, scatter_list=None, src=self.grid[self.dp_rank,self.wp_y_rank,self.wp_x_rank,self.pp_rank,0],group=self.sp_group)
        return input_res

    def sp_sample_gather(self, buffer_id):
        ws_y, ws_x = self.window_size
        wc_y_tot = 720//ws_y
        wc_x_tot = 1440//ws_x
        wc_y_loc = (wc_y_tot//self.WP_Y)
        wc_x_loc = (wc_x_tot//self.WP_X)
        wc_local = wc_y_loc*wc_x_loc
        #sending torch.Size([288, 75, 69])
        inputs = torch.zeros((wc_local, ws_y*ws_x//self.SP, self.model_out_channels), dtype=self.data_dtype, device=self.device)
        torch.distributed.recv(inputs, self.prev_rank, group=self.pp_comm_group_prev)

        torch.distributed.barrier(group=self.sp_group)
        ws_y, ws_x = self.window_size
        wc_y_tot = 720//ws_y
        wc_x_tot = 1440//ws_x
        wc_y_loc = (wc_y_tot//self.WP_Y)
        wc_x_loc = (wc_x_tot//self.WP_X)
        wc_local = wc_y_loc*wc_x_loc
        if self.sp_rank==0:
            #batch_size, (seq//self.SP), dim
            input_list = [torch.zeros((wc_local, ws_y*ws_x//self.SP, self.model_out_channels), dtype=inputs.dtype, device=inputs.device) for _ in range(self.SP)]
        else:
            input_list = None
        
        torch.distributed.gather(inputs, gather_list=input_list, dst=self.grid[self.dp_rank,self.wp_y_rank,self.wp_x_rank,self.pp_rank,0], group=self.sp_group)
        time.sleep(0.05)#Somehow required for correctness on Aurora...
        
        if self.sp_rank==0:
            inputs_out = torch.stack(input_list,dim=1).clone()

            s = self.SP//4
            inputs_out = rearrange(inputs_out, "b (w s) n c -> b c w (s n)", s=s)# n is tokens left from window after SP division. normally here we have:(b wc_y wc_x) (ws_y  ws_x) c
            inputs_out = rearrange(inputs_out, "(b wc_y wc_x) c (d e) (ws_y_h ws_x_h) -> b c (wc_y d ws_y_h) (wc_x e ws_x_h)", wc_y=wc_y_loc, wc_x=wc_x_loc, d=2, e=2, ws_y_h=ws_y//2, ws_x_h=ws_x//2)
            self.pipe_buffers["results"][buffer_id] = inputs_out

    def grid_init(self):
        WP_Y = self.WP_Y
        WP_X = self.WP_X
        PP = self.PP
        SP = self.SP
        assert WP_Y*WP_X*PP*SP <= self.world_size, "Not enough ranks to place parallelisms to"
        DP = self.world_size//(WP_Y*WP_X*PP*SP)
        #assert DP==1, "no DP inference implemented"
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
            self.next_rank=self.grid[self.dp_rank, self.wp_y_rank,self.wp_x_rank, 0, self.sp_rank]
        if self.is_first_stage():
            self.prev_rank=self.grid[self.dp_rank, self.wp_y_rank,self.wp_x_rank, -1, self.sp_rank]
            #Do not shift in stage 0 to stage 1
            self.next_rank = self.grid[self.dp_rank, self.wp_y_rank,self.wp_x_rank, (self.pp_rank+1)%PP, self.sp_rank]
        if self.pp_rank == 1:
            #Do not shift in stage 0 to stage 1
            self.prev_rank = self.grid[self.dp_rank, self.wp_y_rank,self.wp_x_rank, (self.pp_rank-1)%PP, self.sp_rank]

        if self.rank in self.grid[self.dp_rank, :, :, 0, 0]:
            self.res_gather_group = torch.distributed.new_group(self.grid[self.dp_rank, :, :, 0, 0].flatten().tolist(),use_local_synchronization=True)
            self.res_gather_group_head = self.grid[self.dp_rank, 0, 0, 0, 0].item()
            torch.distributed.all_reduce(torch.tensor(0, device=self.device),group=self.res_gather_group)
        torch.distributed.barrier()

        if self.rank in self.grid[:, 0, 0, 0, 0]:
            self.head_dp_group = torch.distributed.new_group(self.grid[:, 0, 0, 0, 0].flatten().tolist(),use_local_synchronization=True)
            torch.distributed.all_reduce(torch.tensor(0, device=self.device),group=self.head_dp_group)
        torch.distributed.barrier()

        self.sp_group = torch.distributed.new_group(self.grid[self.dp_rank, self.wp_y_rank, self.wp_x_rank, self.pp_rank, :].flatten().tolist(),use_local_synchronization=True)
        torch.distributed.all_reduce(torch.tensor(0, device=self.device),group=self.sp_group)
        torch.distributed.barrier()

        weight_ranks = self.grid[:, :, :, self.pp_rank, :].flatten().tolist()
        self.weight_group_size = len(weight_ranks)
        self.weight_group = torch.distributed.new_group(weight_ranks,use_local_synchronization=True)
        torch.distributed.all_reduce(torch.tensor(0, device=self.device),group=self.weight_group)
        torch.distributed.barrier()

        self.pp_comm_group_next = None
        self.pp_comm_group_prev = None

        def init_prev():
            if self.prev_rank != None:
                self.pp_comm_group_prev = torch.distributed.new_group([self.prev_rank, self.rank],use_local_synchronization=True)
                torch.distributed.all_reduce(torch.tensor(0, device=self.device),group=self.pp_comm_group_prev)
            torch.distributed.barrier()

        def init_next():
            if self.next_rank != None:
                self.pp_comm_group_next = torch.distributed.new_group([self.rank,self.next_rank],use_local_synchronization=True)
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
        return inputs
    
    def load(self, buffer_id):
        torch.distributed.barrier(self.sp_group)
        inputs = self.sp_sample_shard(self.model_in,buffer_id)#X

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
            else:
                ape_shape = (1, 69, 721, 1440)
            ape_fn = self.model.ape
            ape = ape_fn(torch.zeros(ape_shape, device=self.device, dtype=self.data_dtype))
            self.model.ape_generated = self.filter_window(ape)

        if self.is_first_stage():
            inputs = inputs.to(self.device, dtype=self.data_dtype).detach().clone()
            self.pipe_buffers["inputs"][buffer_id] = inputs

        #To avoid hiding uneven dataloading. Without this barrier the delays will show up in the next sync
        #In this case that would be the next fwd pass, to be exact the SP all-to-all
        torch.distributed.barrier(self.sp_group)

    def fwd(self, buffer_id):
        with torch.no_grad():
            if self.is_first_stage():
                inputs = self.pipe_buffers["inputs"][buffer_id]
                interval = None
                
                if self.diffusion:
                    interval = self.interval.unsqueeze(0).unsqueeze(0)
                    assert interval.shape == (1,1), interval.shape
                elif self.rit:
                    interval = self.interval
                    assert interval.shape == (1,1), interval.shape
                
                x = \
                    self.model(
                        self.pipe_buffers["inputs"][buffer_id],
                        None,
                        sigma_data=self.sigma_data,
                        condition=None,
                        auxilary=None,
                        interval=interval
                    )
                if self.rit or self.diffusion:
                    x, dt = x
                    assert dt.shape == self.recv_buf_dt.shape, (dt.shape, self.recv_buf_dt.shape)
                    self.pipe_buffers["dt"][buffer_id] = dt
                self.pipe_buffers["outputs"][buffer_id] = x
            else:
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

        #self.cur_nimg += 1

    def send(self, buffer_id):
        outputs = self.pipe_buffers['outputs'][buffer_id]
        assert not torch.isnan(outputs).any(), "trying to send NaN"
        torch.distributed.send(outputs, self.next_rank, group=self.pp_comm_group_next)
        if self.rit or self.diffusion:
            torch.distributed.barrier(group=self.pp_comm_group_next)
            torch.distributed.send(self.pipe_buffers["dt"][buffer_id], self.next_rank, group=self.pp_comm_group_next)

    def send_result(self, buffer_id):
        outputs = self.pipe_buffers["outputs"][buffer_id]
        assert not torch.isnan(outputs).any(), "trying to send NaN"
        torch.distributed.send(outputs, self.next_rank, group=self.pp_comm_group_next)
        
    def recv(self, buffer_id):
        ws_y, ws_x = self.window_size
        wc_y_tot = 720//ws_y
        wc_x_tot = 1440//ws_x
        wc_y_loc = (wc_y_tot//self.WP_Y)
        wc_x_loc = (wc_x_tot//self.WP_X)
        #print(self.rank, "receiving", self.recv_buf.shape, "from", self.prev_rank, flush=True)
        #print("receiving recv_buf_x.shape", self.recv_buf_x.shape, flush=True)
        dims = self.recv_buf_x.shape
        torch.distributed.recv(self.recv_buf_x, self.prev_rank, group=self.pp_comm_group_prev)
        inputs = self.recv_buf_x
        
        self.pipe_buffers['inputs'][buffer_id] = inputs
        
        if self.rit or self.diffusion:
            torch.distributed.barrier(group=self.pp_comm_group_prev)
            torch.distributed.recv(self.recv_buf_dt, self.prev_rank, group=self.pp_comm_group_prev)
            self.pipe_buffers["dt"][buffer_id] = self.recv_buf_dt.detach().clone()


        #self.shift_up_next = self.pp_rank%2 == 1
        #torch.distributed.recv(self.recv_buf_x, self.prev_rank, group=self.pp_comm_group_prev)
        #self.pipe_buffers['inputs'][buffer_id] = self.recv_buf_x.detach().clone()

    
    def load_checkpoint(self, load_ema=False, extra_name=""):
        #TODO
        fname = os.path.join(self.checkpoint_path, f"{extra_name}checkpoint_PP{self.pp_rank}.pth")
        #fname = sorted(glob(os.path.join(self.checkpoint_path, f"*checkpoint_PP{self.pp_rank}.pth")))[-1]
        try:
            checkpoint = torch.load(fname,map_location="cpu",weights_only=True)
        except:
            if self.rank == 0:
                print("Could not find a checkpoint from", self.checkpoint_path, flush=True)
            exit()
            return
        if load_ema:
            self.model.load_state_dict(checkpoint['ema_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.cur_nimg = checkpoint['cur_nimg']
        #self.main_model.load_state_dict(checkpoint['main_model_state_dict'])
        torch.distributed.barrier()
        if self.rank == 0:
            print("loaded a checkpoint from", self.checkpoint_path, flush=True)
        return

    _INSTRUCTION_MAP = {
        schedule.LoadMicroBatch: load,
        schedule.ForwardPass: fwd,
        schedule.SendActivation: send,
        schedule.RecvActivation: recv,
        ReturnResult: sp_sample_gather,
        SendResult: send_result
    }

    _STR_INSTR_MAP = {
        schedule.LoadMicroBatch: "load_data",
        schedule.ForwardPass: "fwd",
        schedule.SendActivation: "send_act",
        schedule.RecvActivation: "recv_act",
        ReturnResult: "return_result",
        SendResult: "send_result",
    }

    def setup_schedule(self):
        self.schedule = EnsembleInferenceSchedule(micro_batches=self.n_members, stages=self.PP, stage_id=self.pp_rank)
    
    def exec_schedule(self, x_next_list, t_hat, condition, debug=False, batch_size=1, interval=None):
        #self.x_next_list = x_next_list#to predict, noisy image
        if (self.rit or self.diffusion) and self.is_first_stage():
            self.interval = interval.to(self.device)
        self.batch_inference = debug or self.diffusion
        self.model_in = condition#the input
        if self.batch_inference:
            schedule = EnsembleInferenceSchedule(micro_batches=batch_size, stages=self.PP, stage_id=self.pp_rank)
        else:
            schedule = self.schedule
        if self.is_first_stage() and self.sp_rank==0:
            assert len(condition) > 0
        self.ensemble = 0
        timers = {}
        result = []
        for step_cmds in schedule:
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
        
        report_timings = self.cur_nimg == (self.n_members*4) or self.cur_nimg == (self.n_members*5)
        #report_timings = False
        if report_timings and self.rank in self.grid[0,0,0,:,0].flatten().tolist():
            print(self.pp_rank, "times:")
            for key in timers:
                print(self.pp_rank, key, timers[key])
            print("",flush=True)

        #gc.collect()
        return timers, self.pipe_buffers["results"]

    def setup_buffers(self, dim, seq, batch_size):
        num_buffers = self.schedule.num_pipe_buffers()
        self.pipe_buffers = {
            'inputs': [],  # batch input and received activations
            'outputs': [],  # activations
            'sigma': [],  # activations
            't': [],  # activations
            'results': [],  # activations
            'dt': [],  # activations
        }
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_buffers)
        #self.recv_buf = torch.empty((1, 32, 4), device=self.device, dtype=torch.bfloat16)
        #self.grad_recv = torch.empty((1, 32, 4), device=self.device, dtype=torch.bfloat16)
        self.recv_buf_x = torch.zeros((batch_size, (seq//self.SP), dim), device=self.device, dtype=torch.bfloat16)
        self.recv_buf_dt = torch.zeros((1, dim), device=self.device, dtype=torch.bfloat16)


    def warmup(self, seq, batch_size, dim):
        if not self.is_first_stage() and not self.is_last_stage():
            in_x = torch.zeros((batch_size, (seq//self.SP), dim), device=self.device, dtype=self.local_model_dtype)
            in_t = torch.zeros((1, dim), device=self.device, dtype=self.local_model_dtype)
            in_dt = torch.zeros((1, dim), device=self.device, dtype=self.local_model_dtype)
            out = self.model(in_x,in_t,in_dt,self.sp_group)