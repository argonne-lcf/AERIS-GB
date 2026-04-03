# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.

# AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions
# This work is licensed under the MIT License. See LICENSE for details.

##TODO

from aeris.parallelism.sequence_parallel import _SeqAllToAll

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from einops import rearrange

if torch.xpu.is_available():
    import intel_extension_for_pytorch
    import oneccl_bindings_for_pytorch

# ----------------------------------------------------------------------------
# Utility Functions

def pair(t):
    return (t, t) if isinstance(t, int) else t

def get_swin_flop_count(
    img_dim: list[int],
    B: int,  # global_batch_size
    l: int,  # num_layers
    c: int,  # num_channels
    d: int,  # hidden_size
    d_ffnn: int,  # ffn_hidden_size
    p: list[int],  # patch_dim
    window_size: list[int],  # iterable of size 2, int
) -> int:
    """Compute the flop counts of the model"""
    img_h, img_w = img_dim
    p_dim = p[0] * p[1]  # patch dim
    L_w = window_size[0] * window_size[1]  # seq length per window
    assert img_h % (window_size[0] * p[0]) == 0
    assert img_w % (window_size[1] * p[1]) == 0
    N_w = img_h * img_w / L_w / p_dim  # num windows per batch
    Bw = B * N_w  # total num windows

    pre_and_post_process = 2 * Bw * p_dim * c * d
    QKVO = 4 * Bw * L_w * d**2
    FA = 2 * Bw * L_w**2 * d
    GLU = 3 * Bw * L_w * d_ffnn * d  # fused (Gate + Up Proj) + Down Proj
    fwd_flop = (QKVO + FA + GLU) * l + pre_and_post_process

    return 6 * fwd_flop  # 3x for fwd+bwd, 2x for mult+add operations

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

    def __init__(self, dim, hidden_dim, norm, rit=False):
        super().__init__()
        self.rit = rit

        self.norm_type = norm
        if not self.rit:
            self.norm = nn.RMSNorm(dim)
        self.w1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        if self.norm_type=="pre":
            if not self.rit:
                x = self.norm(x)
        gate, up_proj = self.w1(x).chunk(2, dim=-1)
        out = self.w2(F.silu(gate) * up_proj)
        if self.norm_type=="post":
            if not self.rit:
                out = self.norm(out)
        return out

class LatentEmbedding(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.cfg = cfg

        self.w1 = nn.Linear(dim, dim, bias=True)
        self.w1.is_time_emb=True
        self.w2 = nn.Linear(dim, dim, bias=True)
        self.w2.is_time_emb=True

    def forward(self, emb):
        out = F.silu(self.w2(F.silu(self.w1(emb))))
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
        xdtype = x.dtype  # b, h, n, d

        x = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = self.cache[None, None, :, :, :]

        x = torch.stack(
            [
                x[..., 0] * rope_cache[..., 0] - x[..., 1] * rope_cache[..., 1],
                x[..., 1] * rope_cache[..., 0] + x[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        )
        x = x.flatten(3)
        return x.to(xdtype)

class SPAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, norm, attn_fn, rit=False, diffusion=False, **rope_kwargs):
        super().__init__()
        self.SP = 12 #TODO get as a variable
        inner_dim = head_dim * heads
        self.global_heads = heads#Local heads
        self.local_heads = heads//self.SP#Local heads
        self.norm_type = norm
        self.rit = rit
        self.diffusion = diffusion

        if not rit:
            self.norm = nn.RMSNorm(dim)

        self.rope = RoPE2D(**rope_kwargs) #TODO implement parallel RoPE

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)
        if attn_fn == "flash":
            self.attn_fn = self.optimized_attention
        if attn_fn == "naive":
            self.attn_fn = self.naive_attention
        if attn_fn == "cosine":
            self.attn_fn = self.cosine_thing
    
    def cosine_thing(self, q, k, v):
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out
    
    def naive_attention(self, q, k, v):
        attn = (q * self.scale) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out
    
    def optimized_attention(self,q,k,v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)


    def forward(self, x, sp_group, benchmark=False):
        #b n (h d)
        
        if self.norm_type=="pre":
            if not (self.rit or self.diffusion):
                x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.global_heads), qkv)
        gather_idx = 1
        scatter_idx = 2
        batch_dim_idx = 0
        start = time.time()
        #torch.distributed.barrier(sp_group)

        seq_gathered_q = _SeqAllToAll.apply(sp_group, q, scatter_idx, 
                                            gather_idx, batch_dim_idx)
        seq_gathered_k = _SeqAllToAll.apply(sp_group, k, scatter_idx, 
                                            gather_idx, batch_dim_idx)
        seq_gathered_v = _SeqAllToAll.apply(sp_group, v, scatter_idx, 
                                            gather_idx, batch_dim_idx)
        
        seq_gathered_q = rearrange(seq_gathered_q, "b n h d -> b h n d")
        seq_gathered_k = rearrange(seq_gathered_k, "b n h d -> b h n d")
        seq_gathered_v = rearrange(seq_gathered_v, "b n h d -> b h n d")
        wh, ww = self.rope.window_size

        
        seq_gathered_q = rearrange(seq_gathered_q, "b h (wc_y wc_x ws_y ws_x) d -> b h (wc_y ws_y wc_x ws_x) d", wc_y=2, wc_x=2, ws_y=wh//2, ws_x=ww//2)#Reorder sequence due to WP+SP
        seq_gathered_k = rearrange(seq_gathered_k, "b h (wc_y wc_x ws_y ws_x) d -> b h (wc_y ws_y wc_x ws_x) d", wc_y=2, wc_x=2, ws_y=wh//2, ws_x=ww//2)#Reorder sequence due to WP+SP
        seq_gathered_v = rearrange(seq_gathered_v, "b h (wc_y wc_x ws_y ws_x) d -> b h (wc_y ws_y wc_x ws_x) d", wc_y=2, wc_x=2, ws_y=wh//2, ws_x=ww//2)#Reorder sequence due to WP+SP
            
        q, k = self.rope(seq_gathered_q), self.rope(seq_gathered_k)
        
        attn_out = self.attn_fn(seq_gathered_q,seq_gathered_k,seq_gathered_v)

        attn_out = rearrange(attn_out, "b h n d -> b n h d")
        attn_out = rearrange(attn_out, "b (wc_y ws_y wc_x ws_x) h d -> b (wc_y wc_x ws_y ws_x) h d", wc_y=2, wc_x=2, ws_y=wh//2, ws_x=ww//2)#Reorder sequence due to WP+SP


        attn_out = _SeqAllToAll.apply(sp_group, attn_out, gather_idx, 
                                            scatter_idx, batch_dim_idx)

        out = rearrange(attn_out, "b n h d -> b n (h d)")
        out = self.wo(out)
        if self.norm_type=="post":
            if not (self.rit or self.diffusion):
                x = self.norm(x)
        return out

class ModulationLinear(nn.Module):
    def __init__(
        self,
        cfg,
        dim,
        out_dim,
        bias,
        layerwise_t_emb
    ):
        super().__init__()
        if layerwise_t_emb:
            self.modulation = nn.Sequential(
                nn.Linear(dim, dim, bias=bias),
                nn.SiLU(),
                nn.Linear(dim, out_dim, bias=bias)
            )
        else:
            self.modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, out_dim, bias=bias)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modulation(x)



# ----------------------------------------------------------------------------
# Swin Transformer Class

class ParallelSwinInputStage(nn.Module):
    def __init__(
            self,
            cfg,
            model_in_channels, 
            dim,
            auxilary_dim=0,  # dimension of auxilary input, 0 = none
            data_dtype=torch.float32,
            model_dtype=torch.bfloat16,
            rit=False,
            diffusion=False,
            emb_norm=False,
            layerwise_t_emb=False,
    ):
        super().__init__()
        self.rit = rit
        self.diffusion = diffusion
        self.cfg = cfg
        self.model_in_channels = model_in_channels
        self.proj = nn.Linear(model_in_channels, dim).to(data_dtype)
        self.proj.is_emb=True
        self.auxilary_embed = nn.Linear(auxilary_dim, dim).to(data_dtype) if auxilary_dim else None
        if auxilary_dim:
            raise NotImplementedError
        self.latent_embed = LatentEmbedding(cfg,dim).to(data_dtype)
        self.layerwise_t_emb = layerwise_t_emb
        if self.diffusion:
            self.latent_embed.is_time_emb=True
        elif self.rit:
            if emb_norm:
                self.interval_embed = nn.Sequential(nn.Linear(1, dim).to(data_dtype),nn.RMSNorm(dim).to(data_dtype))
                self.interval_embed[0].is_emb=True
                self.interval_embed[0].is_time_emb=True
            else:
                self.interval_embed = nn.Linear(1, dim).to(data_dtype)
                self.interval_embed.is_emb=True
                self.interval_embed.is_time_emb=True
        self.ape = PositionalEncoding2D(model_in_channels).to(data_dtype)
        self.auxilary_dim = auxilary_dim
        self.data_dtype = data_dtype
        self.model_dtype = model_dtype
    
    def forward(self, x, sigma=None, sigma_data=1.0, condition=None, auxilary=None, inference=False, interval=None):
        #x is b*w n c torch.Size([30, 700, 200])
        x = x + self.ape_generated  # ??? shape
        x = self.proj(x) # b*w n c-> b*w n d
        if self.diffusion:
            t = timestep_embedding(interval, x.size(2), max_period=self.cfg.model.sinusoidal_emb_max_period)

            assert t.size(0) == 1 and len(t.shape) == 3, t.shape
            if self.layerwise_t_emb:
                t = t[0]
            else:
                t = self.latent_embed(t[0])
            return x.to(self.model_dtype), t.to(self.model_dtype)
        elif self.rit:
            assert interval.shape == (1,1), interval.shape
            t = self.interval_embed(interval)
            return x.to(self.model_dtype), t.to(self.model_dtype)
        else:
            return x.to(self.model_dtype)

class ParallelSwinLayer(nn.Module):
    def __init__(
        self,
        cfg,
        dim,
        heads,
        head_dim,
        norm,
        mlp_dim,
        window_size,
        rope_base,
        model_dtype,
        attn_fn,
        sublayers=1,
        rit=False,
        diffusion=False,
        layerwise_t_emb=False
    ):
        super().__init__()

        self.window_size = window_size
        self.rit = rit
        self.diffusion = diffusion
        self.layerwise_t_emb = layerwise_t_emb
        self.cfg = cfg

        rope_kwargs = {
            "window_size": self.window_size,
            "rope_dim": head_dim // 2,
            "rope_base": rope_base,
        }
        
        self.sublayers = sublayers
        if sublayers > 1 or self.rit or self.diffusion:
            if self.rit or self.diffusion:
                out_dim = 6*dim
                self.modulelist = nn.ModuleList([nn.ModuleList([
                        nn.RMSNorm(dim),
                        SPAttention(dim, heads, head_dim, norm, attn_fn, rit=self.rit, diffusion=self.diffusion, **rope_kwargs),
                        nn.RMSNorm(dim),
                        FeedForward(dim, mlp_dim, norm, rit=True),
                        ModulationLinear(cfg, dim, out_dim, bias=True, layerwise_t_emb=self.layerwise_t_emb)
                    ])for _ in range(sublayers)])
            else:
                self.modulelist = nn.ModuleList([nn.ModuleList([
                        SPAttention(dim, heads, head_dim, norm, attn_fn, **rope_kwargs),
                        FeedForward(dim, mlp_dim, norm)])
                    for _ in range(sublayers)])
        else:
            self.attn = SPAttention(dim, heads, head_dim, norm, attn_fn, **rope_kwargs)
            self.ff = FeedForward(dim, mlp_dim, norm)

    def forward(self, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, sp_group, benchmark=False) -> torch.Tensor:
        if self.cfg.model.layerwise_t_emb_after_mod_norm:
            for i, l in enumerate(self.modulelist):
                n1, attn, n2, ff, mod = l
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod(dt).chunk(6, dim=1)
                #x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
                #x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
                x = x + gate_msa.unsqueeze(1) * attn(n1(modulate(x, shift_msa, scale_msa)), sp_group, benchmark)
                x = x + gate_mlp.unsqueeze(1) * ff(n2(modulate(x, shift_mlp, scale_mlp)))
            return x

        if self.rit or self.diffusion:
            for i, l in enumerate(self.modulelist):
                n1, attn, n2, ff, mod = l
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod(dt).chunk(6, dim=1)
                #x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
                #x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
                x = x + gate_msa.unsqueeze(1) * attn(modulate(n1(x), shift_msa, scale_msa), sp_group, benchmark)
                x = x + gate_mlp.unsqueeze(1) * ff(modulate(n2(x), shift_mlp, scale_mlp))
            return x
        
        if self.sublayers > 1:
            for i, l in enumerate(self.modulelist):
                attn, ff = l
                xp = x
                x = attn(x, sp_group, benchmark)  # num_windows * b, n, d
                x = xp + x
                x = x + ff(x)
            return x
        else:
            xp = x
            x = self.attn(x, sp_group, benchmark)  # num_windows * b, n, d
            x = xp + x
            return x + self.ff(x)



class ParallelSwinOutputStage(nn.Module):
    def __init__(self, cfg, dim, model_out_channels, norm, data_dtype=torch.float32, model_dtype=torch.float32, rit=False, diffusion=False, layerwise_t_emb=False):
        super().__init__()
        self.data_dtype = data_dtype
        self.norm_type = norm
        self.rit = rit
        self.cfg = cfg
        self.diffusion = diffusion
        self.layerwise_t_emb = layerwise_t_emb
        self.norm = nn.RMSNorm(dim).to(data_dtype)  # b, n, d
        self.norm.is_head = True
        if rit or diffusion:
            if self.layerwise_t_emb:
                self.modulation = nn.Sequential(
                    nn.Linear(dim, dim, bias=True),
                    nn.SiLU(),
                    nn.Linear(dim, 2 * dim, bias=True)
                )
            else:
                self.modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(dim, 2 * dim, bias=True)
                )
        self.head = nn.Linear(dim, model_out_channels, bias=False).to(data_dtype)  # b, n, d -> b (h w) c
        #b (h w) c

    def forward(self, x, t, dt, sp_group):
        #print("output_head_in", x.shape, flush=True)
        #torch.Size([1, 32, 384])
        x = x.to(self.data_dtype)
        #t = t.to(self.data_dtype)
        #x = self.norm(x, t)
        #x = self.norm(x, t)
        if self.rit or self.diffusion:
            dt = dt.to(self.data_dtype)
            shift, scale = self.modulation(dt).chunk(2, dim=1)
            x = modulate(self.norm(x), shift, scale)
        else:
            x = self.norm(x)
        out = self.head(x)

        #torch.zeros(out.shape, dtype=out.dtype, device=out.device)
        return out


def init_layer(
        engine,
        cfg,
        model_in_channels: int,
        model_out_channels: int,
        input_channels: int,
        condition_channels: int,
        window_size: tuple[int, int],  # number of patches in a window
        patch_size=(16, 16),
        dim=512,
        heads=12,
        head_dim=64,
        mlp_dim=512,
        rope_base=10_000,
        data_dtype=torch.float32,
        model_dtype=torch.bfloat16,
        norm_type="pre",
        attn_fn="flash",
        sublayers=1,
        rit=False,
        diffusion=False,
        emb_norm=False,
        layerwise_t_emb=False
    ):
        #img_resolution = image_height, image_width = pair(img_resolution)
        patch_size = pair(patch_size)
        window_size = pair(window_size)

        if engine.is_first_stage():
            return ParallelSwinInputStage(cfg, model_in_channels, dim, data_dtype=data_dtype, model_dtype=model_dtype, rit=rit, diffusion=diffusion, emb_norm=emb_norm, layerwise_t_emb=layerwise_t_emb)
        elif engine.is_last_stage():
            return ParallelSwinOutputStage(cfg, dim, model_out_channels, norm_type, data_dtype=data_dtype, model_dtype=model_dtype, rit=rit, diffusion=diffusion, layerwise_t_emb=layerwise_t_emb)
        else:
            return ParallelSwinLayer(cfg, dim, heads, head_dim, norm_type, mlp_dim, window_size, rope_base, model_dtype=model_dtype, attn_fn=attn_fn, sublayers=sublayers, rit=rit, diffusion=diffusion, layerwise_t_emb=layerwise_t_emb)
