from mpi4py import MPI  # isort:skip

import argparse
import os
import numpy as np
import h5py
import torch

from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from era5 import ERA5HDF5InferenceDataset, AttributeSubset, AERIS_SP_ERA5_Data
from model import LocalAERIS, convert_inference_checkpoint

from torch.distributed.device_mesh import DeviceMesh
import torch.accelerator as acc

parser = argparse.ArgumentParser()
# general args
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
parser.add_argument("--steps", type=int, default=8, help="Number of prediction steps")
parser.add_argument("--diffusion-steps", type=int, default=20, help="Number of prediction steps")
parser.add_argument("--stride", type=int, default=1000, help="Number of prediction steps")
parser.add_argument("--start-sample", type=int, default=-1, help="Starting sample")
parser.add_argument("--end-sample", type=int, default=-1, help="Number of samples use")
parser.add_argument("--members", type=int, default=1, help="Number of samples use")
parser.add_argument("--sequence-parallel", type=int, default=1, help="Shard images over N GPUs")
parser.add_argument("--WP-Y", type=int, default=1, help="Shard images over Y direction over N GPUs")
parser.add_argument("--WP-X", type=int, default=1, help="Shard images over X direction over N GPUs")
parser.add_argument("--S_churn", type=float, default=0.00, help="S_churn")
parser.add_argument("--S_min", type=float, default=1, help="S_min")
parser.add_argument("--S_max", type=float, default=1.53, help="S_max")
parser.add_argument("--S_noise", type=float, default=3.0, help="S_noise")
parser.add_argument("--sigma_max", type=float, default=-1, help="sigma_max")
parser.add_argument("--sigma_min", type=float, default=-1, help="sigma_min")

def rollout(
    model,
    device,
    mesh,
    cfg,
    aeris_sp_data,
    local_indices,
    members: int,
    steps: int,
    diffusion_steps: int,
    S_churn: int,
    S_min: int,
    S_max: int,
    S_noise: int,
    sigma_min: float,
    sigma_max: float,
    save_data = False
):
    sp_rank = mesh.get_local_rank(mesh_dim=1)
    sp_group = mesh.get_group(mesh_dim=1)
    SP = torch.distributed.get_world_size(group=sp_group)
    DP = torch.distributed.get_world_size()//SP
    rank = mesh.get_rank()

    sigma_data = cfg.model.sigma_data

    rollout_steps = steps
    in_channels = 70+74
    out_channels = 70
    
    sigma_min = cfg.model.sigma_min if sigma_min == -1 else sigma_min
    sigma_max = cfg.model.sigma_max if sigma_max == -1 else sigma_max

    ramp = torch.linspace(0, 1, diffusion_steps, device=device)
    rho = 10
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    t_steps = torch.atan(sigmas / sigma_data)
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    X = None
    n_local_ic = len(local_indices)

    generator = torch.Generator(device=device)
    generator.manual_seed(rank)

    if save_data and rank==0:
        assert DP == 1, "data saving not yet implemented with data parallelism"
        write_f = h5py.File("/flare/datascience/vhat/terraflux/aeris_gb/outputs/out.hdf5", 'w')
        dset_write = write_f.create_dataset("input", (2, n_local_ic, rollout_steps+1,720, 1440, out_channels), 'f')
        dset_write = write_f["input"]
    
    gp = aeris_sp_data.get_gp().to(device)
    lsm = aeris_sp_data.get_lsm().to(device)

    if rank == 0:
        print(f"starting generation")
    for i in range(0,n_local_ic):
        if rank == 0:
            print(f"generating ic:{i+1}/{n_local_ic}, at index {local_indices[i]} on dp {rank}", flush=True)

        X, rad_T, X_orig = aeris_sp_data.get() #labels: B,S,C,...

        first_frame = aeris_sp_data.gather_tensor(X.cpu())#1st member

        X = X.clone()
        rad_T = rad_T.to(device)
        X_un = aeris_sp_data.unstandardize_x(X.numpy(force=True), 0, out_channels, channel_dim=2)

        if save_data:
            rollouts = np.ones((rollout_steps, members, 720*1440//SP, out_channels), dtype=np.float32)
            inputs_saved = np.ones((720*1440//SP, in_channels), dtype=np.float32)
            gathered = aeris_sp_data.gather_tensor(torch.tensor(X_un, device="cpu"))
            if rank == 0:
                dset_write[:, i, 0] = gathered.repeat(2,1,1).view(2,720,1440,out_channels)
            
        for member in range(members):
            if rank==0:
                print(f"member:{member+1}/{members}", flush=True)
            
            X_un = aeris_sp_data.unstandardize_x(X.numpy(force=True), 0, out_channels, channel_dim=2)
            condition = X.clone().to(device)
            torch.xpu.synchronize()
            for rollout_step in range(rollout_steps):
                if rank==0:
                    print(f"rollout step:{rollout_step+1}/{rollout_steps}", flush=True)
                condition_concat = torch.cat([condition, rad_T[rollout_step:rollout_step+1], gp, lsm, rad_T[rollout_step+1:rollout_step+2]], dim=2)

                latents=torch.randn(X.shape, generator=generator, device=device)
                #latents=torch.zeros(X.shape, device=device) #debugging setting for less randomness
                x_t = latents * sigma_data

                for diffusion_step in range(diffusion_steps):
                    
                    """churn = S_churn > 0
                    if churn:
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
                        z = S_noise * torch.randn(x_t.shape, generator=generator, device=device)
                        x_t = cos_sl * x_t + sin_sl * z
                    else:"""
                    gamma = 0
                    s, t = t_steps[diffusion_step], t_steps[diffusion_step + 1]
                    
                    delta = t - s

                    if rank==0:
                        print(f"diffusion step:{diffusion_step+1}/{diffusion_steps} {s} {t} {gamma}", flush=True)
                    model_in = torch.cat([x_t / sigma_data, condition_concat], dim=2).to(device)
                    if i==0 and member == 0 and diffusion_step == 0 and save_data:
                        inputs_saved[:] = model_in[0].cpu() #(720*1440//SP, in_channels)

                    # Euler
                    with torch.no_grad():
                        F_s = model(model_in.to(device), s.view(1))
                    
                    x_euler = x_t + delta * sigma_data * F_s
                    
                    # second-order Heun correction
                    if diffusion_step < diffusion_steps - 1:
                        model_in = torch.cat([x_euler / sigma_data, condition_concat], dim=2) 

                        with torch.no_grad():
                            F_t = model(model_in.to(device), t.view(1))
                        x_t = x_t + delta * sigma_data * 0.5 * (F_s + F_t)
                    else:
                        x_t = x_euler

                Y_out = x_t
                Y_un = aeris_sp_data.unstandardize_t(Y_out.numpy(force=True), 0, out_channels, channel_dim=2)
                X_out = X_un + Y_un
                X_un = X_out
                
                
                condition=torch.tensor(aeris_sp_data.standardize_x(X_un, 0, out_channels, channel_dim=2), device=condition.device, dtype=condition.dtype)

                if save_data:
                    rollouts[rollout_step,member] = X_un[0].copy()
        if save_data:
            first_member = aeris_sp_data.gather_tensor(torch.tensor(rollouts[:,0],device="cpu"))#1st member
            ens_mean = aeris_sp_data.gather_tensor(torch.tensor(rollouts.mean(axis=1),device="cpu"))#ens mean
            if rank == 0:
                #dset_write Shape: (2, n_local_ic, rollout_steps+1,720, 1440, out_channels)
                #rollouts Shape: (rollout_steps, members, 720*1440//SP, out_channels)
                dset_write[0,i,1:] = first_member.view(-1, 720,1440,out_channels)
                dset_write[1,i,1:] = ens_mean.view(-1, 720,1440,out_channels)
                write_f.flush()

def main(args):
    cfg = OmegaConf.load(os.path.join(args.checkpoint, ".hydra", "config.yaml"))
    rank = int(MPI.COMM_WORLD.Get_rank())
    world_size = int(MPI.COMM_WORLD.Get_size())
    device_count = acc.device_count()
    local_rank = rank % device_count
    acc.set_device_index(local_rank)
    device = torch.device(f"{acc.current_accelerator()}:{local_rank}")
    # Call the init process
    backend = "xccl" if str(acc.current_accelerator())=="xpu" else "nccl"
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    SP = args.sequence_parallel

    assert world_size%SP == 0, "need to be able to divide ranks with SP"
    DP = world_size//SP
    wp_dims = (args.WP_Y,args.WP_X)
    use_wp = wp_dims[0] > 1 or wp_dims[1] > 1
    if use_wp:
        assert SP==wp_dims[0]*wp_dims[1], "SP needs to equal WP if WP is enabled"
    mesh = DeviceMesh(str(acc.current_accelerator()), [[i+j*SP for i in range(SP)] for j in range(DP)])# [DP,SP]
    dp_rank = mesh.get_local_rank(mesh_dim=0)
    sp_rank = mesh.get_local_rank(mesh_dim=1)
    sp_group = mesh.get_group(mesh_dim=1)
    sp_group_ranks = torch.distributed.get_process_group_ranks(sp_group)
    sp_group_gloo = torch.distributed.new_group(sp_group_ranks,backend="gloo",use_local_synchronization=True)
    torch.distributed.barrier()
    #initialize communicator to have p2p comms working later with XCCL
    torch.distributed.all_reduce(torch.tensor(1).to(device),group=sp_group)

    #path = "/eagle/datascience/vhat/gb25_cli/enchanced2_merged_test.hdf5"
    #root = "/eagle/MDClimSim/tungnd/data/wb2/0.25deg_1_step_6hr_h5df_fix_bug/"
    path = "/flare/SAFS/vhat/data/enchanced2_merged_test.hdf5"
    root = "/flare/datasets/wb2/0.25deg_1_step_6hr_h5df_fix_bug/"

    variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200", "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500", "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "geopotential_1000", "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "u_component_of_wind_1000", "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "v_component_of_wind_1000", "temperature_50", "temperature_100", "temperature_150", "temperature_200", "temperature_250", "temperature_300", "temperature_400", "temperature_500", "temperature_600", "temperature_700", "temperature_850", "temperature_925", "temperature_1000", "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200", "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500", "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925", "specific_humidity_1000", "sea_surface_temperature" ,"toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask"]
    dataset = ERA5HDF5InferenceDataset(variables, root, path, interval=cfg.model.interval, rollout_count=args.steps)
    torch.manual_seed(0)

    if rank==0:
        print("initializing and loading model...", flush=True)

    heads = cfg.model.heads
    dim = cfg.model.dim
    head_dim = cfg.model.head_dim
    mlp_dim = cfg.model.mlp_dim
    heads = cfg.model.heads
    window_size = cfg.model.window_size
    sublayers=cfg.model.sublayers
    sinusoidal_emb_max_period=cfg.model.sinusoidal_emb_max_period
    n_layers=cfg.model.PP_stages
    
    model = LocalAERIS(
        device_mesh=mesh,
        heads=heads,
        dim=dim,
        head_dim=head_dim,
        mlp_dim=mlp_dim,
        window_size=window_size,
        image_shape=(720,1440),
        rope_base=10_000,
        sublayers=sublayers,
        sinusoidal_emb_max_period=sinusoidal_emb_max_period,
        n_layers=n_layers,
        model_in_channels=72*2,
        model_out_channels=70,
        SP=SP,
        sp_rank=sp_rank,
        wp_dims=wp_dims
    ).to(device)

    assert (model.image_shape[0]*model.image_shape[1])%SP == 0, "can't divide image with SP"
    if use_wp:
        assert model.image_shape[0]%wp_dims[0] == 0, "can't divide image height with WP"
        assert model.image_shape[1]%wp_dims[1] == 0, "can't divide image width with WP"
    else:
        assert model.heads%SP == 0, "can't divide heads with SP"
    
    convert_inference_checkpoint(args.checkpoint, cfg.model.PP_stages, model, map_location=device)

    model.input_stage.ape_generated = model.input_stage.ape(torch.zeros((1, 72*2, 721, 1440), dtype=model.input_stage.data_dtype, device=device))[:,:,1:,:]
    
    if SP>1:
        prev_ape = model.input_stage.ape_generated
        if use_wp:
            WP_grid = np.arange(SP).reshape((wp_dims))
            my_y,my_x = tuple(i.item() for i in np.where(WP_grid==sp_rank))
            wp_y, ws_y, wp_x, ws_x = wp_dims[0], model.window_size[0], wp_dims[1], model.window_size[1]
            prev_ape = rearrange(prev_ape, "b c (wp_y wcl_y ws_y) (wp_x wcl_x ws_x) -> b wp_y wp_x (wcl_y ws_y wcl_x ws_x) c", wp_y=wp_y, ws_y=ws_y, wp_x=wp_x, ws_x=ws_x)
            new_ape = prev_ape[:,my_y,my_x].contiguous()
        else:
            raise NotImplementedError
            prev_ape = rearrange(prev_ape, "b c h w -> b (h w) c")
            new_ape = prev_ape.chunk(SP,dim=1)[sp_rank].contiguous()
        model.input_stage.ape_generated = new_ape
    else:
        raise NotImplementedError
        model.input_stage.ape_generated = rearrange(model.input_stage.ape_generated, "b c h w -> b (h w) c")

    stride = args.stride
    
    start = 0 if args.start_sample == -1 else args.start_sample
    end = 1464-((args.steps)*(cfg.model.interval//6)) if args.end_sample == -1 else args.end_sample+1

    global_ic = end - start

    stride_indices = list(np.arange(1463)[start:end:stride])
    assert len(stride_indices) % DP == 0, f"data amount not dividable to DP evenly (requirement for local eval), {len(stride_indices)}%{DP} != 0"

    indices = stride_indices[dp_rank::DP]
    
    dataset = AttributeSubset(dataset, indices=indices)
    if rank==0:
        print("len dataset", len(dataset), len(indices), flush=True)

    if sp_rank == 0:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=2,
            prefetch_factor=8,
            persistent_workers=True,
        )
    else:
        dataloader = None
    
    aeris_sp_data = AERIS_SP_ERA5_Data(dataset,dataloader,70,args.steps,mesh,sp_group_gloo,wp_dims=wp_dims,image_shape=model.image_shape, window_size=model.window_size, device=device)

    torch.distributed.barrier()
    if rank==0:
        print("done setting up data", flush=True)
    
    with torch.no_grad():
        rollout(
            model,
            device,
            mesh,
            cfg,
            aeris_sp_data,
            local_indices=indices,
            members=args.members,
            steps=args.steps,
            diffusion_steps=args.diffusion_steps,
            S_churn=args.S_churn,
            S_min=args.S_min,
            S_max=args.S_max,
            S_noise=args.S_noise,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            save_data=False
        )

    if rank==0:
        print("done", flush=True)
    torch.distributed.barrier()
    exit()
    #Hangs on Aurora
    #if torch.distributed.is_initialized():
    #    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
