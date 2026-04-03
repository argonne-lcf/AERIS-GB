# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.

# AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions
# This work is licensed under the MIT License. See LICENSE for details.

from mpi4py import MPI  # isort:skip , needed import for polaris
import os
import shutil
import warnings
from glob import glob
from typing import Tuple
import time
import ezpz
import hydra
import numpy as np
import psutil
import torch
import torch.distributed
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler
from torchinfo import summary

from aeris.data.samplers import InfiniteSampler
from aeris.utils import io
from aeris.parallelism.parallel_engine import ParallelEngine

try:
    import wandb  # needs cli login
except (ImportError, ModuleNotFoundError):
    wandb = None


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)


def resume_setup(cfg: DictConfig) -> Tuple[DictConfig, str]:
    # TODO: do we need to broadcast file IO?
    if cfg.resume is None:
        return cfg, None

    run_dir = cfg.resume
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    checkpoints = sorted(glob(os.path.join(run_dir, "checkpoints", "checkpoint*.pt")))
    assert os.path.isdir(run_dir), FileNotFoundError(f"{run_dir} is not a directory")
    assert checkpoints, FileNotFoundError(
        f"No checkpoints in {os.path.join(run_dir, 'checkpoints')}"
    )
    ckpt = checkpoints[-1]  # latest checkpoint

    if ezpz.get_rank() == 0:
        shutil.copytree(
            os.path.join(run_dir, ".hydra"),
            os.path.join(os.getcwd(), ".hydra"),
            dirs_exist_ok=True,
        )

    io.log0(f"Resuming from {ckpt}")
    return cfg, ckpt


@hydra.main(version_base=None, config_name="train")
def main(cfg: DictConfig):

    _ = ezpz.setup_torch(backend="ddp")

    np.random.seed((cfg.seed * ezpz.get_world_size() + ezpz.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))

    io.log0("Before data barriers...")
    torch.distributed.barrier()
    io.log0("Middle of data barriers...")

    #dataset: Dataset = instantiate(cfg.data.dataset, _convert_="object")
    from aeris.data.era5 import ERA5NRMRTIDataset, ERA5NRMDataset
    path = cfg.data.root
    root = "/flare/datasets/wb2/0.25deg_1_step_6hr_h5df_fix_bug/"
    prefetch_factor = (4 if cfg.data.data_workers > 0 else None)
    if cfg.model.benchmark and cfg.model.rit:
        path = os.path.join(path, f"y_{cfg.model.WP_Y}_x_{cfg.model.WP_X}")
        prefetch_factor = 8
        cfg.data.data_workers = 8
        dataset = ERA5NRMRTIDataset(root, path)
    elif cfg.model.diffusion:
        test = path == "/flare/SAFS/vhat/data/nonres-30x30-enchanced2_test/"#Hack to debug training on test set
        if cfg.model.benchmark:
            raise NotImplementedError
        path = os.path.join(path, f"y_{cfg.model.WP_Y}_x_{cfg.model.WP_X}")
        prefetch_factor = 8
        dataset = ERA5NRMDataset(root, path, enhanced_channels3_train=cfg.model.enhanced_channels3, enhanced_channels4_train=cfg.model.enhanced_channels4, interval=cfg.model.interval, test=test)
    else:
        raise NotImplementedError
    
    torch.distributed.barrier()
    io.log0("After data barrier...")

    ws = cfg.model.window_size #y,x (h w)

    img_size = (720,1440)
    #dataset.img_resolution
    from models.parallel_swin import init_layer
    init_model_fn = init_layer
    init_model_kwargs={}

    if cfg.model.enhanced_channels4:
        init_model_kwargs["condition_channels"] = dataset.n_channels + 1 #for the T in diffusion, inputs
        init_model_kwargs["input_channels"] = dataset.n_channels - 3 #for the T+1 in diffusion, label
        init_model_kwargs["model_in_channels"] = 2*dataset.n_channels - 2
        init_model_kwargs["model_out_channels"] = dataset.n_channels - 3
    elif cfg.model.enhanced_channels3:
        init_model_kwargs["condition_channels"] = dataset.n_channels + 1 #for the T in diffusion, inputs
        init_model_kwargs["input_channels"] = dataset.n_channels - 3 #for the T+1 in diffusion, label
        init_model_kwargs["model_in_channels"] = 2*dataset.n_channels - 2
        init_model_kwargs["model_out_channels"] = dataset.n_channels - 3
    elif cfg.model.diffusion:
        init_model_kwargs["condition_channels"] = dataset.n_channels
        init_model_kwargs["input_channels"] = dataset.n_channels
        init_model_kwargs["model_in_channels"] = 2*dataset.n_channels
        init_model_kwargs["model_out_channels"] = dataset.n_channels
    else:
        init_model_kwargs["condition_channels"] = 0
        init_model_kwargs["input_channels"] = dataset.n_channels
        init_model_kwargs["model_out_channels"] = dataset.n_channels
        init_model_kwargs["model_in_channels"] = dataset.n_channels

    init_model_kwargs["window_size"] = ws
    init_model_kwargs["patch_size"] = (1,1)
    init_model_kwargs["rope_base"] = 10_000
    init_model_kwargs["data_dtype"] = torch.float32
    init_model_kwargs["model_dtype"] = torch.bfloat16 
    init_model_kwargs["dim"] = cfg.model.dim#768, 4608
    init_model_kwargs["heads"] = cfg.model.heads#12, 36
    init_model_kwargs["head_dim"] = cfg.model.head_dim#32, 128
    init_model_kwargs["mlp_dim"] = cfg.model.mlp_dim#1024, 18432
    init_model_kwargs["norm_type"] = cfg.model.norm
    init_model_kwargs["attn_fn"] = cfg.model.attn_fn
    init_model_kwargs["sublayers"] = cfg.model.sublayers
    init_model_kwargs["rit"] = cfg.model.rit
    init_model_kwargs["diffusion"] = cfg.model.diffusion
    init_model_kwargs["lr_min_factor"] = cfg.trainer.lr_min_factor
    init_model_kwargs["layerwise_t_emb"] = cfg.model.layerwise_t_emb
    lr_rampup_img = cfg.trainer.lr_rampup_kimg*1000
    ema_halflife_img= cfg.trainer.ema_halflife_kimg*1000
    total_img = cfg.trainer.total_kimg*1000
    #print("cfg.model_parallel", cfg.model_parallel, flush=True)
    PP = cfg.model.PP_stages
    SP = cfg.model.SP
    WP_X = cfg.model.WP_X
    WP_Y = cfg.model.WP_Y
    GAS = cfg.data.gas
    SEQ = ws[0]*ws[1]
    assert init_model_kwargs["heads"] % SP == 0, "num_heads must be divisible by SP"
    global_windows = (img_size[0]//ws[0])*(img_size[1]//ws[1])
    local_windows = global_windows//(WP_X*WP_Y)
    
    torch.distributed.barrier()
    io.log0("Before model init...")
    #Added args too many times here...
    engine = ParallelEngine(cfg, PP, SP, WP_X, WP_Y, GAS, ezpz.get_rank(), ezpz.get_world_size(), SEQ, local_windows, ezpz.get_torch_device(), init_model_fn, cfg.model.sigma_data ,cfg.model.checkpoint_dir, total_img=total_img, lr_rampup_img=lr_rampup_img, ema_halflife_img=ema_halflife_img, **init_model_kwargs)
    torch.distributed.barrier()
    io.log0("After model init...")
    if cfg.model.dist_optimizer:
        engine.optimizer = instantiate(cfg.optimizer, [engine.local_param_buf_opt], _convert_="object")
    elif (
        cfg.optimizer.get("_target_", "")
        in [
            "torch.optim.Adam",
            "torch.optim.AdamW",
        ]
        and cfg.optimizer.get("weight_decay", 0) != 0
        and cfg.model.custom_decay5 == True
    ):
        decay, no_decay = [], []
        for name, m in engine.main_model.named_parameters():
            # learned embeddings and layernorm (except for modulation in adaLN)
            if (isinstance(m, torch.nn.Linear)) and not hasattr(m, "is_head") and not hasattr(m, "is_emb") and not hasattr(m, "is_time_emb") and not "head" in name and not "proj" in name:
                decay.append(m)
            else:
                no_decay.append(m)
        param_groups = [
            {
                "params": decay,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0,
            },
        ]
        engine.optimizer = instantiate(cfg.optimizer, params=param_groups, _convert_="object")
    elif (
        cfg.optimizer.get("_target_", "")
        in [
            "torch.optim.Adam",
            "torch.optim.AdamW",
        ]
        and cfg.optimizer.get("weight_decay", 0) != 0
        and cfg.model.custom_decay == True
    ):
        decay, no_decay = [], []
        for name, m in engine.main_model.named_parameters():
            # learned embeddings and layernorm (except for modulation in adaLN)
            if "proj" in name or "head" in name or ("norm" in name and "modulation" not in name):
                no_decay.append(m)
            else:
                decay.append(m)
        param_groups = [
            {
                "params": decay,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0,
            },
        ]
        engine.optimizer = instantiate(cfg.optimizer, params=param_groups, _convert_="object")
    else:
        engine.optimizer = instantiate(cfg.optimizer, engine.main_model.parameters(), _convert_="object")
    torch.distributed.barrier()
    io.log0("Start engine warmup...")
    engine.warmup(SEQ, local_windows, cfg.model.dim)

    load = cfg.model.load
    if load:
        engine.load_checkpoint()
    if cfg.model.refresh_decay:
        for group in engine.optimizer.param_groups:
            if "weight_decay" in group and group["weight_decay"] > 0:
                group["weight_decay"] = cfg.optimizer.weight_decay
    io.log0("Entering barrier after checkpoint loading...")
    
    torch.distributed.barrier()
    io.log0("Setup dataset...")
    engine.setup_dataset(dataset, cfg)

    if engine.is_first_stage() or engine.is_last_stage():
        dataset_sampler: Sampler = InfiniteSampler(
            dataset=dataset,
            rank=engine.dp_rank,
            num_replicas=engine.DP,
            shuffle=True,
            seed=cfg.seed+engine.cur_nimg,
        )
        dataloader = iter(
            DataLoader(
                dataset=dataset,
                sampler=dataset_sampler,
                batch_size=1,
                pin_memory=False,
                num_workers=cfg.data.data_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=True,
            )
        )
        engine.setup_data_sampling(dataset_sampler, dataloader)
    #print("setup done", flush=True)
    from aeris.models.parallel_swin import get_swin_flop_count
    flops = get_swin_flop_count(img_size, GAS*engine.DP, (engine.PP-2)*cfg.model.sublayers, dataset.n_channels, cfg.model.dim, cfg.model.mlp_dim, (1,1), ws)

    torch.distributed.barrier()
    if engine.rank == 0 or engine.rank == engine.grid[-1, -1, -1, -1, -1] or (engine.rank in engine.grid[0, 0, 0, :, 0].flatten().tolist()):
        print("start wandb init", flush=True)
        io.log0(OmegaConf.to_yaml(cfg))
        if wandb is not None and not os.environ.get("WANDB_DISABLED", False):
            _ = ezpz.setup_wandb(project_name="aeris", config=cfg)
            #run = wandb.init(
            #    sync_tensorboard=False,
            #    project="aeris",
            #)
        print("done wandb init", flush=True)
        time.sleep(1)
    torch.distributed.barrier()
    if engine.rank==0:
        print("done wandb init barrier, begin training", flush=True)
        time.sleep(1)
    torch.distributed.barrier()

    if engine.benchmark:
        torch.xpu.empty_cache()
    start_step = engine.cur_nimg//(GAS*engine.DP) if cfg.trainer.benchmark_steps < 0 else 0
    end_step = total_img if cfg.trainer.benchmark_steps < 0 else cfg.trainer.benchmark_steps
    for i in range(start_step,end_step):
        start = time.time()
        timers = engine.exec_schedule()
        end = time.time()
        tflops_s_tile = (flops/((end-start)*1e12))/engine.world_size
        lrs = [g["lr"] for g  in engine.optimizer.param_groups]
        peak_mem_gb = ezpz.get_max_memory_allocated(engine.device) / 2**30
        reserved_mem_gb = ezpz.get_max_memory_reserved(engine.device) / 2**20
        if engine.rank==0:
            print(f"{i}, {i*GAS*engine.DP}/{total_img}", "{:.2f}s".format(end-start), "{:.2f} TFLOP/s/tile".format(tflops_s_tile), f"lr:{lrs}", flush=True)
            #print(f"lr:{lrs}",f"cur_nimg:{engine.cur_nimg}",f"lr_rampup_img:{engine.lr_rampup_img}",f"total_img:{engine.total_img}",f"base_lr:{engine.base_lr}", flush=True)
        if engine.benchmark and engine.rank==12:
            print("alloc: {:.1f}GB, res: {:.1f}GB".format(peak_mem_gb, reserved_mem_gb), flush=True)

        def module_norm(module):
            # Iterate through all parameters in the model
            total_l2_norm_squared = torch.tensor(0.0)
            for param in module.parameters():
                total_l2_norm_squared += torch.linalg.vector_norm(param.detach(), ord=2).pow(2).cpu()

            # Calculate the final L2 norm by taking the square root of the sum of squared L2 norms
            return torch.sqrt(total_l2_norm_squared)
        
        if wandb is not None and not os.environ.get("WANDB_DISABLED", False) and not os.environ.get("WANDB_MODE", "enabled") == "disabled":
            metrics = {
                "train/iter": i,
                "train/nimg": engine.cur_nimg,
                "train/tflops_s_tile": tflops_s_tile,
                "train/dt/iter": end-start,
                "train/mem/cpu": psutil.Process(os.getpid()).memory_info().rss / 2**30,
            }

            if engine.rank == 0:
                wandb.log(metrics, step=i)

            if engine.rank in engine.grid[0, 0, 0, :, 0].flatten().tolist():
                metrics = {f"train/stage_norm/stage_norm_{engine.pp_rank}":engine.stage_norm}
                if not engine.is_first_stage() and not engine.is_last_stage() and engine.diffusion and engine.layerwise_t_emb:
                    #module = engine.model.modulelist[0][4].modulation[0]
                    #print("weight_norm0", engine.pp_rank, module_norm(module), flush=True)
                    #for i in range(cfg.model.sublayers):
                    i = 0
                    if (engine.pp_rank == engine.PP-2 or engine.pp_rank==1):
                        norm_0 = module_norm(engine.model.modulelist[i][4].modulation[0])
                        norm_2 = module_norm(engine.model.modulelist[i][4].modulation[2])
                        metrics[f"train/weight_norm/stage_{engine.pp_rank}/{i}_modulation_0"] = norm_0
                        metrics[f"train/weight_norm/stage_{engine.pp_rank}/{i}_modulation_2"] = norm_2
                        print(f"weight_norm/stage_{engine.pp_rank}/{i}_modulation", "{:.2f} {:.2f}".format(norm_0, norm_2), flush=True)
                    #module = engine.model.modulelist[0][4].modulation[2]
                    #print("weight_norm2", engine.pp_rank, module_norm(module), flush=True)
                if cfg.trainer.report_timings:
                    for key in timers:
                        metrics[f"train/pp_timers/{key}/rank_{engine.pp_rank}"] = timers[key] 
                wandb.log(metrics, step=i)
            
            if engine.rank == engine.grid[-1, -1, -1, -1, -1]:
                metrics = {
                    "train/loss": engine.tot_loss,
                    "train/total_norm": engine.total_norm,
                    "lr": lrs[0],
                }
                wandb.log(metrics, step=i)
    
    if cfg.trainer.sleep_after_done:
        time.sleep(100000)
    
    print("done")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    exit()


if __name__ == "__main__":
    main()
