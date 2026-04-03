# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.

# AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions
# This work is licensed under the MIT License. See LICENSE for details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
import numpy as np
import math
import os
from einops import rearrange

# ----------------------------------------------------------------------------
# Utility Functions

def swin_shift(tensor, shift_up, mesh, img_shape, window_size, wp_dims):
    wcl_y = img_shape[0]//window_size[0]//wp_dims[0]
    wcl_x = img_shape[1]//window_size[1]//wp_dims[1]
    ws_y, ws_x = window_size[0], window_size[1]
    assert ws_y == 60
    assert ws_x == 60
    if len(tensor.shape) == 3:
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == wcl_y*ws_y*wcl_x*ws_x, tensor.shape
        assert tensor.shape[2] == 1536, tensor.shape
    else:
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == wcl_y*ws_y*wcl_x*ws_x, tensor.shape
        assert tensor.shape[2] == 12, tensor.shape
        assert tensor.shape[2] == 1536, tensor.shape
    original_len = len(tensor.shape)
    
    sp_group = mesh.get_group(mesh_dim=1)
    sp_rank = mesh.get_local_rank(mesh_dim=1)
    if original_len == 3:
        tensor = rearrange(tensor, "b (wcl_y ws_y wcl_x ws_x) d -> b (wcl_y ws_y) (wcl_x ws_x) d", wcl_y=wcl_y, ws_y=ws_y, wcl_x=wcl_x, ws_x=ws_x).contiguous()
    else:
        tensor = rearrange(tensor, "b (wcl_y ws_y wcl_x ws_x) h d -> b (wcl_y ws_y) (wcl_x ws_x) h d", wcl_y=wcl_y, ws_y=ws_y, wcl_x=wcl_x, ws_x=ws_x).contiguous()
    rank = torch.distributed.get_rank()
    SP = torch.distributed.get_world_size(group=sp_group)
    wp_y, wp_x = wp_dims[0], wp_dims[1]
    WP_grid = np.arange(SP).reshape((wp_y, wp_x))
    my_coords = np.where(WP_grid==sp_rank)
    my_y,my_x = tuple(i.item() for i in np.where(WP_grid==sp_rank))
    my_coords = (my_y,my_x)

    h = ws_y//2
    w = ws_x//2
    if shift_up:
        corner = tensor[:,:h,:w].contiguous()
        vertical_slice = tensor[:,h:,:w].contiguous()
        horizontal_slice = tensor[:,:h,w:].contiguous()
        corner_src = WP_grid[((my_y+1)%wp_y,(my_x+1)%wp_x)]
        corner_dst = WP_grid[((my_y-1)%wp_y,(my_x-1)%wp_x)]
        vertical_src = WP_grid[(my_y,(my_x+1)%wp_x)]
        vertical_dst = WP_grid[(my_y,(my_x-1)%wp_x)]
        horizontal_src = WP_grid[((my_y+1)%wp_y,my_x)]
        horizontal_dst = WP_grid[((my_y-1)%wp_y,my_x)]
    else:
        corner = tensor[:,-h:,-w:].contiguous()
        vertical_slice = tensor[:,:-h,-w:].contiguous()
        horizontal_slice = tensor[:,-h:,:-w].contiguous()
        corner_src = WP_grid[((my_y-1)%wp_y,(my_x-1)%wp_x)]
        corner_dst = WP_grid[((my_y+1)%wp_y,(my_x+1)%wp_x)]
        vertical_src = WP_grid[(my_y,(my_x-1)%wp_x)]
        vertical_dst = WP_grid[(my_y,(my_x+1)%wp_x)]
        horizontal_src = WP_grid[((my_y-1)%wp_y,my_x)]
        horizontal_dst = WP_grid[((my_y+1)%wp_y,my_x)]

    recv_corner = torch.zeros_like(corner)
    recv_vertical_slice = torch.zeros_like(vertical_slice)
    recv_horizontal_slice = torch.zeros_like(horizontal_slice)

    torch.distributed.barrier(group=sp_group)
    torch.xpu.synchronize()
    #Alternate order to avoid deadlocks
    if my_x%2==0:
        torch.distributed.send(corner, dst=corner_dst, group=sp_group)
        torch.distributed.recv(recv_corner, src=corner_src, group=sp_group)
        torch.distributed.send(vertical_slice, dst=vertical_dst, group=sp_group)
        torch.distributed.recv(recv_vertical_slice, src=vertical_src, group=sp_group)
    else:
        torch.distributed.recv(recv_corner, src=corner_src, group=sp_group)
        torch.distributed.send(corner, dst=corner_dst, group=sp_group)
        torch.distributed.recv(recv_vertical_slice, src=vertical_src, group=sp_group)
        torch.distributed.send(vertical_slice, dst=vertical_dst, group=sp_group)

    #Alternate order to avoid deadlocks
    if my_y%2 == 0:
        torch.distributed.send(horizontal_slice, dst=horizontal_dst, group=sp_group)
        torch.distributed.recv(recv_horizontal_slice, src=horizontal_src, group=sp_group)
    else:
        torch.distributed.recv(recv_horizontal_slice, src=horizontal_src, group=sp_group)
        torch.distributed.send(horizontal_slice, dst=horizontal_dst, group=sp_group)

    torch.xpu.synchronize()
    torch.distributed.barrier(group=sp_group)

    #The received corner is now in the wrong place but will be handled by the roll below
    if shift_up:
        tensor[:,:h,:w] = recv_corner
        tensor[:,h:,:w] = recv_vertical_slice
        tensor[:,:h,w:] = recv_horizontal_slice
    else:
        tensor[:,-h:,-w:] = recv_corner
        tensor[:,:-h,-w:] = recv_vertical_slice
        tensor[:,-h:,:-w] = recv_horizontal_slice

    if shift_up:
        tensor = torch.roll(tensor, shifts=(-h, -w), dims=(1,2))
    else:
        tensor = torch.roll(tensor, shifts=(h, w), dims=(1,2))
    if original_len == 3:
        out = rearrange(tensor, "b (wcl_y ws_y) (wcl_x ws_x) d -> b (wcl_y ws_y wcl_x ws_x) d", wcl_y=wcl_y, ws_y=60, wcl_x=wcl_x, ws_x=60)
    else:
        out = rearrange(tensor, "b (wcl_y ws_y) (wcl_x ws_x) h d -> b (wcl_y wcl_x ws_y ws_x) h d", wcl_y=wcl_y, ws_y=60, wcl_x=wcl_x, ws_x=60)
    return out.contiguous()

def modulate(x, shift, scale):
    if scale == None:
        return x + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000):
    """Sinusoidal timestep embeddings."""
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=t.dtype) / half
    ).to(device=t.device)
    args = t[:, None].to(t.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    embedding = (
        embedding.reshape(embedding.shape[0], 2, -1).flip(1).reshape(*embedding.shape)
    )  # flip sin/cos as done with edm

    return embedding

# ----------------------------------------------------------------------------
# Helper Classes

class FeedForward(nn.Module):
    """SwiGLU FeedForward"""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate, up_proj = self.w1(x).chunk(2, dim=-1)
        out = self.w2(F.silu(gate) * up_proj)
        return out

class PositionalEncoding2D(nn.Module):
    """https://github.com/tatp22/multidim-positional-encoding"""

    def __init__(self, channels, max_positions=10_000):
        super().__init__()
        self.channels = int(math.ceil(channels / 4) * 2)
        inv_freq = 1.0 / (
            max_positions ** (torch.arange(0, self.channels, 2).float() / self.channels)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def _get_emb(self, sin_inp):
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, x):
        assert x.ndim == 4, "input has to be 4d!"
        if self.cached_penc is not None and self.cached_penc.shape == x.shape:
            return self.cached_penc

        b, c, h, w = x.shape
        pos_x = torch.arange(h, device=x.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(w, device=x.device, dtype=self.inv_freq.dtype)
        sin_inp_x = pos_x.unsqueeze(1) * self.inv_freq
        sin_inp_y = pos_y.unsqueeze(1) * self.inv_freq
        emb_x = self._get_emb(sin_inp_x)
        emb_y = self._get_emb(sin_inp_y)

        emb_x = emb_x.unsqueeze(1).expand(h, w, self.channels)
        emb_y = emb_y.unsqueeze(0).expand(h, w, self.channels)

        emb = torch.cat([emb_x, emb_y], dim=-1)
        emb = emb[..., :c].permute(2, 0, 1)
        self.cached_penc = emb.unsqueeze(0).repeat(b, 1, 1, 1).to(x.dtype)
        return self.cached_penc

class RoPE2D(nn.Module):
    """Axial Frequency 2D Rotary Positional Embeddings (https://arxiv.org/pdf/2403.13298).

    The embedding is applied to the x-axis and y-axis separately.
    """

    def __init__(
        self,
        window_size: tuple[int, int],
        rope_dim: int,
        rope_base: int = 10_000,
    ):
        super().__init__()
        self.window_size = window_size
        self.rope_dim = rope_dim
        self.rope_base = rope_base
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.rope_base
            ** (
                torch.arange(0, self.rope_dim, 2)[: (self.rope_dim // 2)].float()
                / self.rope_dim
            )
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache()

    def build_rope_cache(self):
        wh, ww = self.window_size
        patches_per_tile = wh * ww

        patch_idx = torch.arange(
            patches_per_tile, dtype=self.theta.dtype, device=self.theta.device
        )
        patch_x_pos = patch_idx % ww
        patch_y_pos = patch_idx // ww

        x_theta = torch.einsum("i, j -> ij", patch_x_pos, self.theta).float()
        y_theta = torch.einsum("i, j -> ij", patch_y_pos, self.theta).float()

        freqs = torch.cat([x_theta, y_theta], dim=-1)
        cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = x.to("cpu")
        xdtype = x.dtype  # b, h, n, d

        x = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = self.cache[None, None, :, :, :].to("cpu")

        x = torch.stack(
            [
                x[..., 0] * rope_cache[..., 0] - x[..., 1] * rope_cache[..., 1],
                x[..., 1] * rope_cache[..., 0] + x[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        )
        x = x.flatten(3)
        return x.to(xdtype).to(device)

# ----------------------------------------------------------------------------
# Swin Transformer Classes

class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, window_size, image_shape, device_mesh, wp_dims=(1,1), **rope_kwargs):
        super().__init__()
        #self.global_heads = global_heads
        self.heads = heads
        inner_dim = head_dim * heads
        self.device_mesh = device_mesh
        self.window_size = window_size
        self.image_shape = image_shape
        self.ws_y = window_size[0]
        self.ws_x = window_size[1]
        self.wcl_y = image_shape[0]//self.ws_y//wp_dims[0]
        self.wcl_x = image_shape[1]//self.ws_x//wp_dims[1]
        self.wp = wp_dims[0] > 1 or wp_dims[1] > 1
        self.wp_dims = wp_dims
        self.scale = head_dim**-0.5

        self.rope = RoPE2D(window_size, **rope_kwargs)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)

        self.attn_fn = self.naive_attention
    
    def naive_attention(self, q, k, v):
        attn = (q * self.scale) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out
    
    def optimized_attention(self,q,k,v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)

    def forward(self, x):
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (c h d) -> b n h (c d)", c=3, h=self.heads).contiguous()

        qkv = rearrange(qkv, "b (wcl_y ws_y wcl_x ws_x) h d -> (b wcl_y wcl_x) h (ws_y ws_x) d", wcl_y=self.wcl_y, wcl_x=self.wcl_x, ws_y=self.ws_y, ws_x=self.ws_x).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        #q, k = self.rope(q), self.rope(k)
        out = self.attn_fn(q,k,v)

        out = rearrange(out, "(b wcl_y wcl_x) h (ws_y ws_x) d -> b (wcl_y ws_y wcl_x ws_x) (h d)", wcl_y=self.wcl_y, wcl_x=self.wcl_x, ws_y=self.ws_y, ws_x=self.ws_x).contiguous()
        out = self.wo(out)
        return out

class ModulationLinear(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        bias
    ):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.Linear(dim, dim, bias=bias),
            nn.SiLU(),
            nn.Linear(dim, out_dim, bias=bias)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modulation(x)

class InputStage(nn.Module):
    def __init__(
            self,
            model_in_channels, 
            dim,
            sinusoidal_emb_max_period,
            data_dtype=torch.float32,
            model_dtype=torch.bfloat16
    ):
        super().__init__()
        self.model_in_channels = model_in_channels
        self.sinusoidal_emb_max_period = sinusoidal_emb_max_period
        self.data_dtype = data_dtype
        self.model_dtype = model_dtype
        
        self.proj = nn.Linear(model_in_channels, dim).to(data_dtype)

        self.ape = PositionalEncoding2D(model_in_channels).to(data_dtype)
    
    def forward(self, x, interval):
        #x shape:  [B, S, C]
        #interval shape: [B]
        assert x.shape == self.ape_generated.shape
        x = x + self.ape_generated  # ??? shape
        
        x = self.proj(x) # b n c-> b n d
        t = timestep_embedding(interval, x.size(2), max_period=self.sinusoidal_emb_max_period)

        assert t.size(0) == 1 and len(t.shape) == 2, t.shape
        return x.to(self.model_dtype), t.to(self.model_dtype)

class SwinLayer(nn.Module):
    def __init__(
        self,
        device_mesh,
        dim,
        heads,
        head_dim,
        mlp_dim,
        window_size,
        image_shape,
        shifted_grid,
        rope_base,
        wp_dims=(1,1),
        sublayers=1,
    ):
        super().__init__()

        self.window_size = window_size

        rope_kwargs = {
            "window_size": self.window_size,
            "rope_dim": head_dim // 2,
            "rope_base": rope_base,
        }
        
        self.sublayers = sublayers

        out_dim = 6*dim
        self.modulelist = nn.ModuleList([nn.ModuleList([
                RMSNorm(dim),
                Attention(dim, heads, head_dim, image_shape=image_shape, device_mesh=device_mesh, wp_dims=wp_dims, **rope_kwargs),
                RMSNorm(dim),
                FeedForward(dim, mlp_dim),
                ModulationLinear(dim, out_dim, bias=True)
            ])for i in range(sublayers)])

    #def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    def forward(self, x, dt):
        #x: [b, n, d]
        for _, l in enumerate(self.modulelist):
            n1, attn, n2, ff, mod = l
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod(dt).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * attn(n1(modulate(x, shift_msa, scale_msa)))
            x = x + gate_mlp.unsqueeze(1) * ff(n2(modulate(x, shift_mlp, scale_mlp)))
        #print("forward layer", flush=True)
        return x

class OutputStage(nn.Module):
    def __init__(self, dim, model_out_channels, image_shape, data_dtype=torch.float32):
        super().__init__()
        self.data_dtype = data_dtype
        self.image_shape = image_shape
        self.norm = RMSNorm(dim).to(data_dtype)  # b, n, d

        self.modulation = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )

        self.head = nn.Linear(dim, model_out_channels, bias=False).to(data_dtype)  # b, n, d -> b (h w) c
        #b (h w) c

    def forward(self, x, dt):
        x = x.to(self.data_dtype)
        
        dt = dt.to(self.data_dtype)
        shift, scale = self.modulation(dt).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)

        out = self.head(x)
        return out

class LocalAERIS(nn.Module):
    def __init__(
            self,
            device_mesh,
            heads, 
            dim,
            head_dim,
            mlp_dim,
            window_size,
            image_shape,
            rope_base,
            sublayers,
            sinusoidal_emb_max_period,
            n_layers,
            model_in_channels,
            model_out_channels,
            SP=1,
            sp_rank=1,
            wp_dims=(1,1)
        ):
        super().__init__()
        model_dtype=torch.bfloat16
        data_dtype=torch.float32
        self.mesh = device_mesh
        self.image_shape = image_shape
        self.window_size = window_size
        self.wp_dims = wp_dims
        self.heads = heads
        self.input_stage = InputStage(model_in_channels, dim, sinusoidal_emb_max_period).to(data_dtype)
        layers = []
        for i in range(n_layers-2):
            shifted_grid = i % 2 == 1
            subLayer = SwinLayer(device_mesh, dim, heads, head_dim, mlp_dim, window_size, image_shape, shifted_grid, rope_base, sublayers=sublayers, wp_dims=wp_dims)
            layers.append(subLayer)
        self.layers = nn.ModuleList(layers).to(model_dtype)
        self.output_stage = OutputStage(dim, model_out_channels, image_shape).to(data_dtype)

    def forward(self,x, dt):
        out, t = self.input_stage(x, dt)
        for i, l in enumerate(self.layers):
            if i%2 == 1:
                out = swin_shift(out, True, mesh=self.mesh, img_shape=self.image_shape, window_size=self.window_size, wp_dims=self.wp_dims)
            out = l(out,t)
            if i%2 == 1:
                out = swin_shift(out, False, mesh=self.mesh, img_shape=self.image_shape, window_size=self.window_size, wp_dims=self.wp_dims)
        out = self.output_stage(out, t)
        return out

def convert_inference_checkpoint(path, N, model, map_location="cpu"):
    new_state_dict = {}
    checkpoint = torch.load(os.path.join(path, f"checkpoint_PP{0}.pth"), map_location=map_location, weights_only=True)["ema_state_dict"]
    for key in checkpoint.keys():
        if "latent_embed" not in key:#Remove latent_embed that was accidentally left in the checkpoint
            new_state_dict[f"input_stage.{key}"] = checkpoint[key]
    for i in range(0,N-2):
        checkpoint = torch.load(os.path.join(path, f"checkpoint_PP{i+1}.pth"), map_location=map_location, weights_only=True)["ema_state_dict"]
        for key in checkpoint.keys():
            if ("1.norm" not in key):#Remove norm that was accidentally left in the checkpoint
                new_state_dict[f"layers.{i}.{key}"] = checkpoint[key]
    checkpoint = torch.load(os.path.join(path,f"checkpoint_PP{N-1}.pth"), map_location=map_location, weights_only=True)["ema_state_dict"]
    for key in checkpoint.keys():
        new_state_dict[f"output_stage.{key}"] = checkpoint[key]

    assert model.state_dict().keys() == new_state_dict.keys(), f"loaded state dict does not match 1:1 to model, something missing? {model.state_dict().keys()}, {state_dict.keys()}"
    model.load_state_dict(new_state_dict)
    #return new_state_dict