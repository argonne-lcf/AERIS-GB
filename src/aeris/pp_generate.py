# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.

# AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions
# This work is licensed under the MIT License. See LICENSE for details.

from mpi4py import MPI  # isort:skip
import argparse
import os
import time
import gc
import math
import ezpz
import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from aeris.data.samplers import AttributeSubset
from aeris.utils import io
from aeris.parallelism.sp_inference_engine import InferenceEngine

import h5py
from einops import rearrange

parser = argparse.ArgumentParser()
# general args
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
parser.add_argument("--steps", type=int, default=8, help="Number of prediction steps")
parser.add_argument("--diffusion-steps", type=int, default=20, help="Number of prediction steps")
parser.add_argument("--batch-size", type=int, default=1, help="Number of prediction steps")
parser.add_argument("--stride", type=int, default=1000, help="Number of prediction steps")
parser.add_argument("--start-sample", type=int, default=-1, help="Starting sample")
parser.add_argument("--end-sample", type=int, default=-1, help="Number of samples use")
parser.add_argument("--members", type=int, default=1, help="Number of samples use")
parser.add_argument("--debug", action="store_true", help="debug mode")
parser.add_argument("--local-eval", action="store_true", help="Do eval on the go")
parser.add_argument("--skip-local-eval", action="store_true", help="Skip actual eval on local-eval")
parser.add_argument("--use-ema", action="store_true", help="Use EMA weights instead of non-EMA")
parser.add_argument("--extra-name", type=str, default="", help="Extra to prepend to a name")
parser.add_argument("--extra-outname", type=str, default="", help="Extra to prepend to a name")
parser.add_argument("--all-evals", action="store_true", help="Do all available metrics")
parser.add_argument("--calculate-locally", action="store_true", help="Do not gather images for eval")
parser.add_argument("--save-output", action="store_true", help="Save outputs")
parser.add_argument("--S_churn", type=float, default=0.00, help="S_churn")
parser.add_argument("--S_min", type=float, default=1, help="S_min")
parser.add_argument("--S_max", type=float, default=1.53, help="S_max")
parser.add_argument("--S_noise", type=float, default=3.0, help="S_noise")
parser.add_argument("--sigma_max", type=float, default=-1, help="S_noise")
parser.add_argument("--sigma_min", type=float, default=-1, help="S_noise")
parser.add_argument("--clamp-sst", action="store_true", help="Limit SST values on Land for stability")


# output args

# ----------------------------------------------------------------------------



def d_rollout_local_mp(
    engine,
    odir: str,
    save_output: bool,
    dataloader: DataLoader,
    local_indices,
    stride_indices: int,
    members: int,
    steps: int,
    diffusion_steps: int,
    use_ema: bool,
    enhanced_channels3: bool,
    enhanced_channels4: bool,
    interval: int,
    all_evals: bool,
    skip_local_eval: bool,
    calculate_locally: bool,
    S_churn: int,
    S_min: int,
    S_max: int,
    S_noise: int,
    sigma_min: float,
    sigma_max: float,
    clamp_sst: bool,
    extra_outname: str
):
    
    rollout_steps = steps
    stride_ic = len(stride_indices)
    out_channels = 70 if enhanced_channels4 else 69

    filename = f"output-{stride_ic}i-{steps}s-{interval}h-{members}m-{diffusion_steps}ds{extra_outname}"
    ofile = os.path.join(odir, f"{filename}.hdf5")

    if engine.rank==0 and save_output:
        write_f = h5py.File(ofile, 'w')
        dset_write = write_f.create_dataset("input", (engine.WP_Y,engine.WP_X,stride_ic,rollout_steps+1,members,out_channels,720//engine.WP_Y,1440//engine.WP_X), 'f')
        write_f.close()
    torch.distributed.barrier()
    if engine.is_first_stage() and engine.sp_rank==0 and save_output:
        write_f = h5py.File(ofile, 'a')
        dset_write = write_f["input"]

    torch.distributed.barrier()
    sigma_min = engine.sigma_min if args.sigma_min == -1 else args.sigma_min
    sigma_max = engine.sigma_max if args.sigma_max == -1 else args.sigma_max

    sigma_data = engine.sigma_data

    device = engine.device
    
    ramp = torch.linspace(0, 1, diffusion_steps, device=device)
    rho = 10
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    t_steps = torch.atan(sigmas / sigma_data)
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])


    #b c (wc_y ws_y) (wc_x ws_x)
    i_dataloader = iter(dataloader)

    X = None
    n_local_ic = len(local_indices)
        #(WP_Y,WP_X,ic,steps+1,69,720//engine.WP_Y,1440//engine.WP_X)
        #for s,X_i in enumerate(X_in):
    generator = torch.Generator(device=engine.device)
    generator.manual_seed(engine.rank)

    if calculate_locally:
        w_lat = torch.cos(torch.deg2rad(torch.linspace(-90, 90, 721)))
        w_lat_1dim = w_lat / w_lat.mean()
        w_lat_shape = (out_channels, 720//engine.WP_Y, 1440//engine.WP_X)
        ws_y = engine.window_size[0]
        w_lat_1dim = w_lat_1dim[1:].reshape(720//(ws_y*engine.WP_Y),engine.WP_Y,-1)[:,engine.wp_y_rank]
        w_lat = w_lat_1dim.reshape((1,-1,1)).expand(w_lat_shape)
    else:
        w_lat_shape = (out_channels, 721, 1440)
        w_lat = torch.cos(torch.deg2rad(torch.linspace(-90, 90, 721)))
        w_lat_1dim = w_lat / w_lat.mean()
        w_lat = w_lat_1dim.view((1,-1,1)).expand(w_lat_shape)

    err = torch.zeros((rollout_steps,5,out_channels))
    if (enhanced_channels3 or enhanced_channels4) and engine.is_first_stage() and engine.sp_rank==0:
        gp_ind = dataloader.dataset.variables.index("geopotential_at_surface")
        #gp = dataloader.dataset.standardize_x(dataloader.dataset.ds[0:1,gp_ind,:,:], gp_ind, gp_ind+1)
        #gp = torch.tensor(gp, device=engine.device, dtype=torch.float).repeat(members,1,1,1)

        lsm_ind = dataloader.dataset.variables.index("land_sea_mask")
        #lsm = dataloader.dataset.standardize_x(dataloader.dataset.ds[0:1,lsm_ind,:,:], lsm_ind, lsm_ind+1)
        #lsm = torch.tensor(lsm, device=engine.device, dtype=torch.float).repeat(members,1,1,1)

        rad_ind = dataloader.dataset.variables.index("toa_incident_solar_radiation")

    if engine.rank == 0:
        print(f"starting generation, with clamp_sst:{clamp_sst} to {filename}")
    for i in range(0,n_local_ic):
        if engine.rank in engine.grid[:, 0, 0, 0, 0]:
            print(f"generating ic:{i+1}/{n_local_ic}, at index {local_indices[i]} on dp {engine.dp_rank}", flush=True)

        item_count = members

        if engine.is_first_stage() and engine.sp_rank==0:
            X, labels = next(i_dataloader)
            X_full = X.to(engine.device)
            X = X_full[:,:out_channels]
            X = X.clone().float()

            X_un_0 = dataloader.dataset.unstandardize_x(X.numpy(force=True), 0, out_channels)
            X_un = X_un_0
            if save_output:
                rollout = np.ones((rollout_steps+1, members, out_channels,720//engine.WP_Y,1440//engine.WP_X), dtype=np.float32)
                rollout[0] = torch.tensor(X_un).repeat(members,1,1,1).clone()
            condition=X.clone().repeat(members,1,1,1)
        
        if (enhanced_channels3 or enhanced_channels4)  and engine.is_first_stage() and engine.sp_rank==0:
            #rad = dataloader.dataset.standardize_x(dataloader.dataset.ds[local_indices[i]:local_indices[i]+rollout_steps+1,rad_ind,:,:],rad_ind, rad_ind+1)
            #rad = torch.tensor(rad, device=engine.device, dtype=torch.float)

            #gp = X_full[:,gp_ind:gp_ind+1].repeat(members,1,1,1)
            #lsm = X_full[:,lsm_ind:lsm_ind+1].repeat(members,1,1,1)
            #print("gp.shape, lsm.shape", gp.shape, lsm.shape, flush=True)
            gp = labels[0,0:1,gp_ind:gp_ind+1].numpy()
            gp = torch.from_numpy(dataloader.dataset.standardize_x(gp, gp_ind, gp_ind+1)).to(device=engine.device, dtype=torch.float).repeat(members,1,1,1).float()
            lsm = labels[0,0:1,lsm_ind:lsm_ind+1].numpy()
            lsm = torch.from_numpy(dataloader.dataset.standardize_x(lsm, lsm_ind, lsm_ind+1)).to(device=engine.device, dtype=torch.float).repeat(members,1,1,1).float()
            
        for rollout_step in range(rollout_steps):
            if engine.rank==0:
                print(f"rollout step:{rollout_step+1}/{rollout_steps}", flush=True)
            if engine.is_first_stage() and engine.sp_rank==0:
                latents=torch.randn(condition.shape, generator=generator, device=engine.device)
                x_t = latents * sigma_data


                if enhanced_channels4 or enhanced_channels3:
                    rad_T = labels[0,rollout_step:rollout_step+1,rad_ind:rad_ind+1].numpy()
                    rad_T = torch.from_numpy(dataloader.dataset.standardize_x(rad_T, rad_ind, rad_ind+1)).to(device=engine.device, dtype=torch.float).repeat(members,1,1,1).float()#standardized
                    rad_T1 = labels[0,rollout_step+1:rollout_step+2,rad_ind:rad_ind+1].numpy()
                    #print("rad_T1.shape, rad_T.shape", rad_T1.shape, rad_T.shape, flush=True)
                    rad_T1 = torch.from_numpy(dataloader.dataset.standardize_x(rad_T1, rad_ind, rad_ind+1)).to(device=engine.device, dtype=torch.float).repeat(members,1,1,1).float()#standardized residual

                #print("multiple shapes", X.shape, x_t.shape, rad_T1.shape, condition.shape, rad_T.shape, gp.shape, lsm.shape, flush=True)
            for diffusion_step in range(diffusion_steps):

                churn = False
                churn_form1 = False
                churn_form2 = True
                churn_form3 = False
                churn_form4 = False
                if churn_form1:
                    s_cur, t = t_steps[diffusion_step], t_steps[diffusion_step + 1]
                    # increase noise temporarily
                    gamma = (
                        min(S_churn / diffusion_steps, np.sqrt(2) - 1)
                        if S_min <= torch.tan(s_cur) <= S_max
                        else 0
                    )
                    s = torch.arctan((1 + gamma) * torch.tan(s_cur))
                    s_l = torch.arctan((torch.tan(s)**2 - torch.tan(s_cur)**2).sqrt())
                    cos_sl, sin_sl = torch.cos(s_l), torch.sin(s_l)
                    if engine.is_first_stage() and engine.sp_rank==0:
                        z = S_noise * torch.randn(x_t.shape, generator=generator, device=engine.device)
                        x_t = cos_sl * x_t + sin_sl * z

                elif churn_form2:
                    s_cur, t = t_steps[diffusion_step], t_steps[diffusion_step + 1]
                    # increase noise temporarily
                    gamma = (
                        min(S_churn / diffusion_steps, np.sqrt(2) - 1)
                        if S_min <= torch.tan(s_cur) <= S_max
                        else 0
                    )
                    s = torch.arctan((1 + gamma) * torch.tan(s_cur))
                    s_l = torch.arctan((torch.tan(s) - torch.tan(s_cur)))
                    cos_sl, sin_sl = torch.cos(s_l), torch.sin(s_l)
                    if engine.is_first_stage() and engine.sp_rank==0:
                        z = S_noise * torch.randn(x_t.shape, generator=generator, device=engine.device)
                        x_t = cos_sl * x_t + sin_sl * z
                elif churn_form3:
                    s_cur, t = t_steps[diffusion_step], t_steps[diffusion_step + 1]
                    # increase noise temporarily
                    gamma = (
                        min(S_churn / diffusion_steps, np.sqrt(2) - 1)
                        if S_min <= torch.tan(s_cur) <= S_max
                        else 0
                    )
                    
                    s = torch.arctan((1 + gamma) * torch.tan(s_cur))
                    s_std = (torch.sin(s)**2 - torch.sin(s_cur)**2).sqrt()
                    if engine.is_first_stage() and engine.sp_rank==0:
                        x_t = x_t + s_std * S_noise * torch.randn(x_t.shape, generator=generator, device=engine.device)

                elif churn_form4:
                    s_cur, t = t_steps[diffusion_step], t_steps[diffusion_step + 1]
                    # increase noise temporarily
                    gamma = (
                        min(S_churn / diffusion_steps, np.sqrt(2) - 1)
                        if S_min <= torch.tan(s_cur) <= S_max
                        else 0
                    )
                    
                    s = torch.arctan((1 + gamma) * torch.tan(s_cur))
                    s_std = (torch.sin(s)**2 - torch.sin(s_cur)**2).sqrt()
                    cos_sl, sin_sl = torch.cos(s_std), torch.sin(s_std)
                    if engine.is_first_stage() and engine.sp_rank==0:
                        z = S_noise * torch.randn(x_t.shape, generator=generator, device=engine.device)
                        x_t = cos_sl * x_t + sin_sl * z

                elif churn:
                    s_cur, t = t_steps[diffusion_step], t_steps[diffusion_step + 1]
                    # increase noise temporarily
                    gamma = (
                        min(S_churn / diffusion_steps, np.sqrt(2) - 1)
                        if S_min <= torch.tan(s_cur) <= S_max
                        else 0
                    )
                    s = torch.arctan(torch.tan(s_cur) + torch.tan(gamma * s_cur))
                    if engine.is_first_stage() and engine.sp_rank==0:
                        x_t = x_t + (torch.tan(s)**2 - torch.tan(s_cur)**2).sqrt() * S_noise * torch.randn(x_t.shape, generator=generator, device=engine.device)
                else:
                    gamma = 0
                    s, t = t_steps[diffusion_step], t_steps[diffusion_step + 1]
                delta = t - s

                if engine.rank==0 and rollout_step ==0:
                    print(f"diffusion step:{diffusion_step+1}/{diffusion_steps} {s} {t} {gamma}", flush=True)
                
                if engine.is_first_stage() and engine.sp_rank==0:
                    if enhanced_channels3 or enhanced_channels4:
                        model_in = torch.cat([x_t / sigma_data, condition, rad_T, gp, lsm, rad_T1], dim=1)
                    else:
                        model_in = torch.cat([x_t / sigma_data, condition], dim=1)
                else:
                    model_in = None
                timers, Y_out = engine.exec_schedule(None, None, model_in, True, item_count, s)
                
                if engine.is_first_stage() and engine.sp_rank==0:
                    # Euler
                    F_s = torch.cat(Y_out[:item_count], dim=0)
                    x_euler = x_t + delta * sigma_data * F_s
                else:
                    x_euler = None


                # second-order Heun correction
                if diffusion_step < diffusion_steps - 1:
                    if engine.is_first_stage() and engine.sp_rank==0:
                        if enhanced_channels3 or enhanced_channels4:
                            model_in = torch.cat([x_euler / sigma_data, condition, rad_T, gp, lsm, rad_T1], dim=1)#TODO check dim #latent (69), T+1, cond (69) , T,G,L
                        else:
                            model_in = torch.cat([x_euler / sigma_data, condition], dim=1)#TODO check dim #latent (69), T+1, cond (69) , T,G,L
                    else:
                        model_in = None
                    timers, Y_out = engine.exec_schedule(None, None, model_in, True, item_count, t)
                    if engine.is_first_stage() and engine.sp_rank==0:
                        F_t = torch.cat(Y_out[:item_count], dim=0)
                        x_t = x_t + delta * sigma_data * 0.5 * (F_s + F_t)
                else:
                    x_t = x_euler
        

            if engine.is_first_stage() and engine.sp_rank==0:
                Y_out = x_t
                Y_un = dataloader.dataset.unstandardize_t(Y_out.numpy(force=True), 0, out_channels)
                X_out = X_un + Y_un

                X_un = X_out
                if save_output:
                    rollout[rollout_step+1] = torch.tensor(X_out).clone() #[rollout_steps+1, members, 69,720//engine.WP_Y,1440//engine.WP_X]
                condition=torch.tensor(dataloader.dataset.standardize_x(X_un, 0, out_channels, clamp_sst=clamp_sst),device=X.device, dtype=X.dtype)
            if not skip_local_eval:
                #print("before eval barrier", engine.rank, flush=True)
                torch.distributed.barrier()
                #print("after eval barrier", engine.rank, flush=True)
                #gather results:
                if engine.is_first_stage() and engine.sp_rank==0:
                    X_out = torch.from_numpy(X_out)
                    #w_lat_1dim = w_lat_1dim
                    #X_un_1st = X_out[0]
                    #X_un_mean = X_out.mean(dim=0)
                    #crps_spread = torch.abs(X_out.unsqueeze(2)-X_out.unsqueeze(1)).sum() #(d e c) (wc_y ws_y) (wc_x ws_x) -> ??
                    #crps_err = torch.abs(X_out - label.unsqueeze(1)).sum() #(d e c) (wc_y ws_y) (wc_x ws_x) -> ??
                    if calculate_locally:
                        label = labels[0,rollout_step+1,:out_channels,:,:] #C H W


                        err_1st = (w_lat * (label-X_out[0]) ** 2).sum(dim=(-1,-2))
                        err_mean = (w_lat * (label-X_out.mean(dim=0)) ** 2).sum(dim=(-1,-2))

                        N = members
                        crps_err_abs = (w_lat_1dim.reshape(1,1,-1,1) * torch.abs(X_out-label.unsqueeze(0))).sum(dim=(0,2,3))/N #[M, C, H, W] ->[C]

                        crps_spread = (w_lat_1dim.reshape(1,1,1,-1,1) * torch.abs(X_out.unsqueeze(1)-X_out.unsqueeze(0))) #[M, C, H, W] -> [M, M, C, H, W]
                        crps_spread = crps_spread.sum(dim=(-2, -1))#[M, M, C, H, W] -> [M, M, C]
                        crps_spread = crps_spread.sum(dim=(0,1))/(2*N*(N-1)) #[M, M, C] -> [C]

                        ssr_var = w_lat_1dim.reshape(1,-1,1) * torch.var(X_out, dim=0) #[M, C, H, W] -> #[C, H, W]
                        ssr_spread = ssr_var.sum(dim=(-2,-1)) #[C, H, W] -> [C]
                        #print("start reducing results", flush=True)
                        torch.distributed.all_reduce(err_1st, group=engine.res_gather_group)
                        time.sleep(0.06)
                        torch.distributed.all_reduce(err_mean, group=engine.res_gather_group)
                        time.sleep(0.06)
                        torch.distributed.all_reduce(crps_err_abs, group=engine.res_gather_group)
                        time.sleep(0.06)
                        torch.distributed.all_reduce(crps_spread, group=engine.res_gather_group)
                        time.sleep(0.06)
                        torch.distributed.all_reduce(ssr_spread, group=engine.res_gather_group)
                        ssr_spread = (ssr_spread/(721*1440)).sqrt()
                        time.sleep(0.06)
                        #print("before DP barrier", engine.rank, flush=True)
                        if engine.wp_x_rank == 0 and engine.wp_y_rank == 0:
                            torch.distributed.barrier(group=engine.head_dp_group)
                            #print("start reducing DP results", engine.rank, flush=True)
                            torch.distributed.all_reduce(err_1st, group=engine.head_dp_group)
                            time.sleep(0.06)
                            torch.distributed.all_reduce(err_mean, group=engine.head_dp_group)
                            time.sleep(0.06)
                            torch.distributed.all_reduce(crps_err_abs, group=engine.head_dp_group)
                            time.sleep(0.06)
                            torch.distributed.all_reduce(crps_spread, group=engine.head_dp_group)
                            time.sleep(0.06)
                            torch.distributed.all_reduce(ssr_spread, group=engine.head_dp_group)
                            time.sleep(0.06)
                            err[rollout_step,0] += (err_1st/(721*1440)).cpu()
                            err[rollout_step,1] += (err_mean/(721*1440)).cpu()
                            err[rollout_step,2] += (crps_err_abs/(721*1440)).cpu()
                            err[rollout_step,3] += (crps_spread/(721*1440)).cpu()
                            err[rollout_step,4] += ssr_spread.cpu()
                            t2m_ind = dataloader.dataset.variables.index("2m_temperature")
                            if engine.rank==0:
                                print(f"{(rollout_step+1)*interval}h scores t2m at ic {i+1}/{n_local_ic}:", "{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(torch.sqrt(err[rollout_step,0][t2m_ind]/((i+1)*engine.DP)),
                                                                                                                                                                torch.sqrt(err[rollout_step,1][t2m_ind]/((i+1)*engine.DP)),
                                                                                                                                                                (err[rollout_step,2][t2m_ind]/((i+1)*engine.DP))-((err[rollout_step,3][t2m_ind])/((i+1)*engine.DP)),
                                                                                                                                                            (err[rollout_step,4][t2m_ind]/((i+1)*engine.DP))/torch.sqrt(err[rollout_step,1][t2m_ind]/((i+1)*engine.DP))), flush=True)

                    else:
                        gather_out_all = [torch.zeros_like(X_out) for _ in range(engine.WP_X*engine.WP_X)]if engine.wp_x_rank == 0 and engine.wp_y_rank == 0 else None
                        torch.distributed.gather(X_out, gather_out_all, dst=engine.res_gather_group_head, group=engine.res_gather_group)
                        time.sleep(0.2)

                        label = labels[0,rollout_step+1,:out_channels,:,:]
                        gather_out_label = [torch.zeros_like(label) for _ in range(engine.WP_X*engine.WP_X)] if engine.wp_x_rank == 0 and engine.wp_y_rank == 0  else None
                        torch.distributed.gather(label, gather_out_label, dst=engine.res_gather_group_head, group=engine.res_gather_group)
                        time.sleep(0.2)

                        if engine.wp_x_rank == 0 and engine.wp_y_rank == 0:
                            X_un_gathered = torch.cat(gather_out_all, dim=1)
                            label_gathered = torch.cat(gather_out_label)
                            #print("X_un_1st_gathered.shape", X_un_1st_gathered.shape, X_un_mean_gathered.shape)

                            def reconstruct(output):
                                rearrange_shape = "(d e c) (wc_y ws_y) (wc_x ws_x) -> c (wc_y d ws_y) (wc_x e ws_x)" if len(output.shape) == 3 else "m (d e c) (wc_y ws_y) (wc_x ws_x) -> m c (wc_y d ws_y) (wc_x e ws_x)"
                                output = rearrange( 
                                    output,
                                    rearrange_shape,
                                    ws_y=engine.window_size[0],
                                    ws_x=engine.window_size[1],
                                    d = engine.WP_Y,
                                    e = engine.WP_X,
                                )
                                zero_count = (output == 0).sum()
                                if zero_count > 0:
                                    print("ALERT,", zero_count, "zero values in output", flush=True)
                                output = F.pad(output, (0,0, 1,0, 0,0))
                                return output
                            
                            X_un_gathered = reconstruct(X_un_gathered)
                            X_un_1st_gathered = X_un_gathered[0]
                            X_un_mean_gathered = X_un_gathered.mean(dim=0)
                            label_gathered = reconstruct(label_gathered)
                            
                            #print("label.shape, X_un_1st.shape, X_un_mean.shape", label_gathered.shape, X_un_1st_gathered.shape, X_un_mean_gathered.shape, flush=True)
                            err_1st = (w_lat * (label_gathered-X_un_1st_gathered) ** 2).sum(dim=(-1,-2))
                            err_mean = (w_lat * (label_gathered-X_un_mean_gathered) ** 2).sum(dim=(-1,-2))

                            torch.distributed.all_reduce(err_1st, group=engine.head_dp_group)
                            torch.distributed.all_reduce(err_mean, group=engine.head_dp_group)
                            #[M, C, H, W]
                            if all_evals:
                                N = members
                                crps_err_abs = (w_lat_1dim.view(1,1,-1,1) * torch.abs(X_un_gathered-label_gathered.unsqueeze(0))).sum(dim=(0,2,3))/N #[M, C, H, W] ->[C]

                                crps_spread = (w_lat_1dim.view(1,1,1,-1,1) * torch.abs(X_un_gathered.unsqueeze(0)-X_un_gathered.unsqueeze(1))) #[M, C, H, W] -> [M, M, C, H, W]
                                crps_spread = crps_spread.mean(dim=(-2, -1))#[M, M, C, H, W] -> [M, M, C]
                                crps_spread = crps_spread.sum(dim=(0,1))/(2*N*(N-1)) #[M, M, C] -> [C]

                                ssr_var = w_lat_1dim.view(1,-1,1) * torch.var(X_un_gathered, dim=0) #[M, C, H, W] -> #[C, H, W]
                                ssr_spread = ssr_var.mean(dim=(-2,-1)).sqrt() #[C, H, W] -> [C]

                                #print("entering reduce with rank", engine.rank, flush=True)
                                torch.distributed.all_reduce(crps_err_abs, group=engine.head_dp_group)
                                torch.distributed.all_reduce(crps_spread, group=engine.head_dp_group)
                                torch.distributed.all_reduce(ssr_spread, group=engine.head_dp_group)
                            if engine.rank==0:
                                err[rollout_step,0] += (err_1st/(721*1440)).cpu()
                                err[rollout_step,1] += (err_mean/(721*1440)).cpu()
                                if all_evals:
                                    err[rollout_step,2] += (crps_err_abs/(721*1440)).cpu()
                                    err[rollout_step,3] += crps_spread.cpu()
                                    err[rollout_step,4] += ssr_spread.cpu()
                                t2m_ind = dataloader.dataset.variables.index("2m_temperature")
                                if all_evals:
                                    print(f"{(rollout_step+1)*interval}h scores t2m at ic {i+1}/{n_local_ic}:", "{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(torch.sqrt(err[rollout_step,0][t2m_ind]/((i+1)*engine.DP)),
                                                                                                                                                                    torch.sqrt(err[rollout_step,1][t2m_ind]/((i+1)*engine.DP)),
                                                                                                                                                                    (err[rollout_step,2][t2m_ind]/((i+1)*engine.DP))-((err[rollout_step,3][t2m_ind])/((i+1)*engine.DP)),
                                                                                                                                                                    (err[rollout_step,4][t2m_ind]/((i+1)*engine.DP))/torch.sqrt(err[rollout_step,1][t2m_ind]/((i+1)*engine.DP))))
                                else:
                                    print(f"{(rollout_step+1)*interval}h ensemble mean t2m at ic {i+1}/{n_local_ic}", torch.sqrt(err[rollout_step,1][t2m_ind]/((i+1)*engine.DP)))
                torch.distributed.barrier()      
        if engine.is_first_stage() and engine.sp_rank==0 and save_output:
            loc_idx = local_indices[i]
            dst_idx = list(stride_indices).index(loc_idx)
            print(f"writing results for wp_y:{engine.wp_y_rank}, wp_X:{engine.wp_x_rank}, initial condition {loc_idx} ({i+1}/{len(local_indices)}) to {dst_idx}", flush=True)
            dset_write[engine.wp_y_rank, engine.wp_x_rank, dst_idx] = rollout
            write_f.flush()
            gc.collect()

    if engine.is_first_stage() and engine.sp_rank==0 and save_output:
        write_f.close()
    if engine.rank == 0:

        resfile = os.path.join(odir, f"{filename}_err_aggregate.npy")
        np.save(resfile, err)
        print(f"done generation use_ema:{use_ema} from checkpoint {engine.checkpoint_path} trained for {engine.cur_nimg}", flush=True)
        err_out = torch.zeros((rollout_steps,4,out_channels))
        err_out[:,0] = torch.sqrt(err[:,0]/stride_ic)#RMSE (1st)
        err_out[:,1] = torch.sqrt(err[:,1]/stride_ic)#RMSE (mean)
        err_out[:,2] = ((err[:,2]/stride_ic)-((err[:,3])/stride_ic))#CRPS
        err_out[:,3] = (err[:,4]/stride_ic)/torch.sqrt((err[:,1]/stride_ic))#Spread/Skill
        #if save_output:
        resfile = os.path.join(odir, f"{filename}_err.npy")
        np.save(resfile, err_out)

        print(f"{interval*1}h")

        vars = {
            "t2m ": "2m_temperature",
            "u10m": "10m_u_component_of_wind",
            "v10m": "10m_v_component_of_wind",
            "mslp": "mean_sea_level_pressure",
            "z500": "geopotential_500",
            "t850": "temperature_850",
            "q700": "specific_humidity_700",
            "u850": "u_component_of_wind_850",
            "v850": "v_component_of_wind_850",
        }
        variables_list = dataloader.dataset.variables
        for i in [0,1,3]:
            print(f"{interval*(i+1)}h")
            for key in vars:
                var_ind = variables_list.index(vars[key])
                if key == "q700":
                    scale = 1000
                else:
                    scale = 1
                #print(key, err[interval,0][variables_list.index(vars[key])]*scale, err[interval,1][variables_list.index(vars[key])]*scale)
                if all_evals:
                    print(f"{key}:", "{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(err_out[i,0][var_ind]*scale, err_out[i,1][var_ind]*scale, err_out[i,2][var_ind]*scale, err_out[i,3][var_ind]))
                else:
                    print(f"{key}:",
                        torch.sqrt(err[i,0][var_ind]/stride_ic)*scale, 
                        torch.sqrt(err[i,1][var_ind]/stride_ic)*scale)
        

                #latent (69), T+1, cond (69) , T,G,L

# ----------------------------------------------------------------------------
def run_on_rank0(fn, *args, **kwargs):
    if ezpz.get_rank() == 0:
        fn(*args, **kwargs)
    MPI.COMM_WORLD.Barrier()


def main(args):
    cfg = OmegaConf.load(os.path.join(args.checkpoint, ".hydra", "config.yaml"))
    _ = ezpz.setup_torch(backend="ddp")

    if ezpz.get_rank() == 0:
        io.log0(OmegaConf.to_yaml(cfg))

    np.random.seed((cfg.seed) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))

    io.log0("Loading dataset...")

    if cfg.model.enhanced_channels4:
        from aeris.data.era5 import ERA5NRMDataset
        if cfg.model.window_size == (60,60):
            path = "/flare/SAFS/vhat/data/nonres-60x60-enchanced2-test/"
        else:
            path = "/flare/SAFS/vhat/data/nonres-30x30-enchanced2_test/"
        path = os.path.join(path, f"y_{cfg.model.WP_Y}_x_{cfg.model.WP_X}")
        root = "/flare/datasets/wb2/0.25deg_1_step_6hr_h5df_fix_bug/"
        dataset = ERA5NRMDataset(root, path, enhanced_channels=True, interval=cfg.model.interval, enhanced_channels4_inference=True)
    else:
        raise NotImplementedError

    ws = cfg.model.window_size #y,x (h w)
    img_size = (720,1440)
    #dataset.img_resolution
    from aeris.models.parallel_swin import init_layer
    init_model_fn = init_layer
    init_model_kwargs={}

    if cfg.model.enhanced_channels4:
        init_model_kwargs["condition_channels"] = dataset.n_channels + 1 #for the T in diffusion, inputs
        init_model_kwargs["input_channels"] = dataset.n_channels - 3 #for the T+1 in diffusion, label
        init_model_kwargs["model_in_channels"] = 2*dataset.n_channels - 2
        init_model_kwargs["model_out_channels"] = dataset.n_channels - 3
    
    init_model_kwargs["window_size"] = ws
    init_model_kwargs["patch_size"] = (1,1)
    init_model_kwargs["dim"] = cfg.model.dim#768, 4608
    init_model_kwargs["heads"] = cfg.model.heads#12, 36
    init_model_kwargs["head_dim"] = cfg.model.head_dim#32, 128
    init_model_kwargs["mlp_dim"] = cfg.model.mlp_dim#1024, 18432
    init_model_kwargs["rope_base"] = 10_000
    init_model_kwargs["data_dtype"] = torch.float32
    init_model_kwargs["model_dtype"] = torch.bfloat16 
    init_model_kwargs["norm_type"] = cfg.model.norm
    init_model_kwargs["attn_fn"] = cfg.model.attn_fn
    init_model_kwargs["sublayers"] = cfg.model.sublayers
    init_model_kwargs["rit"] = cfg.model.rit
    init_model_kwargs["diffusion"] = cfg.model.diffusion
    init_model_kwargs["layerwise_t_emb"] = cfg.model.layerwise_t_emb

    lr_rampup_img = cfg.trainer.lr_rampup_kimg*1000
    ema_halflife_img= cfg.trainer.ema_halflife_kimg*1000
    total_img = cfg.trainer.total_kimg*1000
    
    PP = cfg.model.PP_stages
    SP = cfg.model.SP
    WP_X = cfg.model.WP_X
    WP_Y = cfg.model.WP_Y

    batch_size = args.batch_size
    n_members = args.members if args.local_eval else batch_size

    SEQ = ws[0]*ws[1]
    assert init_model_kwargs["heads"] % SP == 0, "num_heads must be divisible by SP"
    global_windows = (img_size[0]//ws[0])*(img_size[1]//ws[1])
    local_windows = global_windows//(WP_X*WP_Y)
    checkpoint_dir = args.checkpoint

    engine = InferenceEngine(cfg, PP, SP, WP_X, WP_Y, n_members, ezpz.get_rank(), ezpz.get_world_size(), SEQ, local_windows, ezpz.get_torch_device(), init_model_fn, 1.0, checkpoint_dir, total_img=total_img, lr_rampup_img=lr_rampup_img, ema_halflife_img=ema_halflife_img, **init_model_kwargs)
    
    engine.warmup(SEQ, local_windows, cfg.model.dim)


    stride = args.stride
    
    start = 0 if args.start_sample == -1 else args.start_sample
    end = 1464-((args.steps)*(cfg.model.interval//6)) if args.end_sample == -1 else args.end_sample+1
    global_ic = end - start
    dp_size = engine.DP
    dp_rank = engine.dp_rank

    stride_indices = list(np.arange(1463)[start:end:stride])
    stride_ic = len(stride_indices)

    #indices = list(range(0,initial_conditions,stride))
    indices = stride_indices[dp_rank::dp_size]
    local_ic = len(indices)
    if args.local_eval:
        assert stride_ic % dp_size == 0, f"data amount not dividable to DP evenly (requirement for local eval), {stride_ic}%{dp_size} != 0"
    
    
    if cfg.model.rit and not cfg.model.diffusion:
        rit_indices = np.arange(global_ic)*3
        ds_indices = list(rit_indices[::stride][dp_rank::dp_size])
        dataset = AttributeSubset(dataset, indices=ds_indices)
    else:
        dataset = AttributeSubset(dataset, indices=indices)
    if ezpz.get_rank()==0:
        print("len dataset", len(dataset), local_ic, flush=True)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=2,
        prefetch_factor=8,
        persistent_workers=True,
    )

    if engine.is_first_stage() or engine.is_last_stage():
        assert cfg.data.wp_load
        engine.dataset=dataset
        engine.wp_load = cfg.data.wp_load
        load_inf_labels = args.steps if args.local_eval else 0
        dataset.open_file(engine.wp_y_rank, engine.wp_x_rank, load_inf_labels=load_inf_labels, load_t=False)


    if engine.rank==0:
        print("done setting up data", flush=True)
    torch.distributed.barrier()
    if engine.rank==0:
        print("done before load_checkpoint barrier", flush=True)

    engine.load_checkpoint(load_ema=args.use_ema, extra_name=args.extra_name)

    io.log0("Setting up output directory/file...")
    io.log0(
        f"{len(dataset)} initials for {args.steps} steps over {1} members"
    )
    odir = os.path.join(args.checkpoint, "output")
    run_on_rank0(os.makedirs, odir, exist_ok=True)

    d_rollout_local_mp(
        engine,
        odir,
        args.save_output,
        dataloader,
        indices,
        stride_indices,
        args.members,
        args.steps,
        args.diffusion_steps,
        args.use_ema,
        cfg.model.enhanced_channels3,
        cfg.model.enhanced_channels4,
        cfg.model.interval,
        args.all_evals,
        args.skip_local_eval,
        args.calculate_locally,
        S_churn = args.S_churn,
        S_min = args.S_min,
        S_max = args.S_max,
        S_noise = args.S_noise,
        sigma_min = args.sigma_min,
        sigma_max = args.sigma_max,
        clamp_sst = args.clamp_sst,
        extra_outname = args.extra_outname
    )

    io.log0("Finished!")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
