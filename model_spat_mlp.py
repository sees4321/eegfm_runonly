
from __future__ import annotations

import json
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EEGModelConfig


def _gather_channel_features(x_ch: torch.Tensor, c_idx: torch.Tensor) -> torch.Tensor:
    B, C, Fdim = x_ch.shape
    B2, L = c_idx.shape
    assert B == B2
    idx = c_idx[..., None].expand(B, L, Fdim)
    return x_ch.gather(dim=1, index=idx)


def build_rope_cache(
    max_pos: int,
    rotary_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert rotary_dim % 2 == 0
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0
    if cos.dim() == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif cos.dim() == 3:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class LegendreAnchorFeatures(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.num_anchors = int(cfg.spatial_qk_num_anchors)
        self.out_dim = int(cfg.spatial_qk_feat_dim)
        self.use_unit = bool(cfg.spatial_bias_use_unit_sphere)
        self.eps = 1e-6

        anchors = torch.randn(self.num_anchors, 3)
        anchors = F.normalize(anchors, p=2, dim=-1)
        self.anchors = nn.Parameter(anchors)

        hidden = cfg.d_model
        self.mlp = nn.Sequential(
            nn.Linear(self.num_anchors, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.out_dim),
        )
        self.norm = nn.LayerNorm(self.out_dim, eps=1e-6)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.use_unit:
            u = coords / coords.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        else:
            u = coords

        a = F.normalize(self.anchors, p=2, dim=-1)
        x = torch.einsum("bcd,rd->bcr", u.float(), a.float()).clamp(-1.0, 1.0)
        # x: (B, N_channels, num_anchors)

        feat = self.mlp(x)       # (B, N_channels, out_dim)
        feat = self.norm(feat)
        return feat.to(coords.dtype)


class CoordMLPEmbedding(nn.Module):
    def __init__(self, d_model: int, coord_jitter_std: float = 0.05, coord_jitter_prob: float = 0.5, w_init: float = 0.0):
        super().__init__()
        self.coord_jitter_std = float(coord_jitter_std)
        self.coord_jitter_prob = float(coord_jitter_prob)
        self.proj = nn.Linear(3, d_model)
        self.emb_w = nn.Parameter(torch.tensor([w_init], dtype=torch.float32))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        B, _, _ = coords.shape
        if self.training and (self.coord_jitter_std > 0) and (self.coord_jitter_prob > 0):
            gate = (torch.rand((B,), device=coords.device) < self.coord_jitter_prob).to(coords.dtype)
            coords = coords + torch.randn_like(coords) * self.coord_jitter_std * gate[:, None, None]
        coords = F.normalize(coords, p=2, dim=-1)
        return self.proj(coords) * self.emb_w.to(dtype=coords.dtype, device=coords.device)


class SpatialBias(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.enabled = str(cfg.spatial_bias_type).lower() not in ("none", "off", "disable", "disabled")
        self.use_unit = bool(cfg.spatial_bias_use_unit_sphere)
        self.scale = float(cfg.spatial_bias_scale)
        self.eps = 1e-6

        if self.enabled:
            hidden = cfg.d_model
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.mlp = None

    def _cosine(self, coords: torch.Tensor) -> torch.Tensor:
        if self.use_unit:
            u = coords / coords.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        else:
            u = coords
        return torch.matmul(u, u.transpose(1, 2)).clamp(-1.0, 1.0)

    def forward(self, coords: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None

        x = self._cosine(coords.float())          # (B, N, N)
        B, N, _ = x.shape

        flat = x.reshape(-1, 1)                   # (B*N*N, 1)
        bias = self.mlp(flat).reshape(B, N, N)    # (B, N, N)

        if self.scale != 1.0:
            bias = bias * self.scale
        return bias


class TimePatchEmbed(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.patch_samples = int(round(cfg.sample_rate * cfg.patch_seconds))
        self.proj = nn.Linear(self.patch_samples, cfg.d_model)

    def forward_packed(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden = int(d_model * mlp_ratio * (2.0 / 3.0))
        self.fc = nn.Linear(d_model, 2 * hidden)
        self.proj = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        x = self.drop(x)
        x = self.proj(x)
        x = self.drop(x)
        return x


class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float,
        rope_theta: float,
        rotary_pct: float,
        spatial_qk_dim: int = 0,
        spatial_qk_scale: float = 1.0,
        use_spatial_qk: bool = False,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = float(attn_dropout)
        self.spatial_qk_scale = float(spatial_qk_scale)

        rotary_dim = int(self.head_dim * rotary_pct)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        self.rotary_dim = max(0, rotary_dim)
        if self.rotary_dim > 0:
            cos, sin = build_rope_cache(
                max_pos=max_seq_len,
                rotary_dim=self.rotary_dim,
                theta=float(rope_theta),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            self.register_buffer("_rope_cos", cos, persistent=False)
            self.register_buffer("_rope_sin", sin, persistent=False)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

        if use_spatial_qk:
            self.spatial_q_proj = nn.Linear(spatial_qk_dim, d_model, bias=False)
            self.spatial_k_proj = nn.Linear(spatial_qk_dim, d_model, bias=False)
        else:
            self.spatial_q_proj = None
            self.spatial_k_proj = None

    def _get_rope(self, rope_pos: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._rope_cos.to(dtype=dtype)
        sin = self._rope_sin.to(dtype=dtype)
        if rope_pos.dim() == 1:
            return cos[rope_pos], sin[rope_pos]
        if rope_pos.dim() == 2:
            return cos[rope_pos], sin[rope_pos]
        raise ValueError(f"rope_pos must be 1D or 2D, got {rope_pos.shape}")

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        rope_pos: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        spatial_q_add: Optional[torch.Tensor] = None,
        spatial_k_add: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rotary_dim > 0:
            if rope_pos is None:
                raise ValueError("rope_pos must be provided when rotary_dim > 0")
            cos, sin = self._get_rope(rope_pos, dtype=q.dtype)
            if self.rotary_dim == self.head_dim:
                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)
            else:
                q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
                k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
                q = torch.cat([apply_rope(q_rot, cos, sin), q_pass], dim=-1)
                k = torch.cat([apply_rope(k_rot, cos, sin), k_pass], dim=-1)

        if (spatial_q_add is not None) or (spatial_k_add is not None):
            if spatial_q_add is None or spatial_k_add is None:
                raise ValueError("spatial_q_add and spatial_k_add must be provided together")
            q_sp = spatial_q_add.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            k_sp = spatial_k_add.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            q = q + self.spatial_qk_scale * q_sp
            k = k + self.spatial_qk_scale * k_sp

        attn_mask = None
        if attn_bias is None:
            if padding_mask is not None:
                attn_mask = (~padding_mask)[:, None, None, :]
        else:
            attn_mask = attn_bias[:, None, :, :] if attn_bias.dim() == 3 else attn_bias
            if attn_mask.dtype != q.dtype:
                attn_mask = attn_mask.to(dtype=q.dtype)
            if padding_mask is not None:
                attn_mask = attn_mask.masked_fill(padding_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)


class CrossAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float, rope_theta: float, rotary_pct: float, max_seq_len: int = 4096):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = float(attn_dropout)

        rotary_dim = int(self.head_dim * rotary_pct)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        self.rotary_dim = max(0, rotary_dim)
        if self.rotary_dim > 0:
            cos, sin = build_rope_cache(
                max_pos=max_seq_len,
                rotary_dim=self.rotary_dim,
                theta=float(rope_theta),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            self.register_buffer("_rope_cos", cos, persistent=False)
            self.register_buffer("_rope_sin", sin, persistent=False)

        self.q = nn.Linear(d_model, d_model, bias=True)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def _get_rope(self, rope_pos: torch.Tensor, dtype: torch.dtype):
        cos = self._rope_cos.to(dtype=dtype)
        sin = self._rope_sin.to(dtype=dtype)
        if rope_pos.dim() == 1:
            return cos[rope_pos], sin[rope_pos]
        if rope_pos.dim() == 2:
            return cos[rope_pos], sin[rope_pos]
        raise ValueError(f"rope_pos must be 1D or 2D, got {rope_pos.shape}")

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        kv_padding_mask: Optional[torch.Tensor],
        rope_pos_q: torch.Tensor,
        rope_pos_k: torch.Tensor,
    ) -> torch.Tensor:
        B, Lq, D = q_in.shape
        _, Lk, _ = kv_in.shape

        q = self.q(q_in).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(kv_in)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rotary_dim > 0:
            cos_q, sin_q = self._get_rope(rope_pos_q, dtype=q.dtype)
            cos_k, sin_k = self._get_rope(rope_pos_k, dtype=q.dtype)
            if self.rotary_dim == self.head_dim:
                q = apply_rope(q, cos_q, sin_q)
                k = apply_rope(k, cos_k, sin_k)
            else:
                q = torch.cat([apply_rope(q[..., : self.rotary_dim], cos_q, sin_q), q[..., self.rotary_dim :]], dim=-1)
                k = torch.cat([apply_rope(k[..., : self.rotary_dim], cos_k, sin_k), k[..., self.rotary_dim :]], dim=-1)

        attn_mask = None
        if kv_padding_mask is not None:
            attn_mask = (~kv_padding_mask)[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out(out)


class FullAttentionBlock(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.use_spatial_bias = bool(cfg.full_attn_use_spatial_bias)
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=1e-6)
        self.attn = MultiheadSelfAttentionRoPE(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_theta=cfg.rope_theta,
            rotary_pct=cfg.rotary_pct,
            spatial_qk_dim=cfg.spatial_qk_feat_dim,
            spatial_qk_scale=cfg.spatial_qk_scale,
            use_spatial_qk=(str(cfg.spatial_qk_type).lower() == "legendre_anchor"),
            max_seq_len=cfg.max_tokens,
        )
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=1e-6)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        rope_pos: torch.Tensor,
        chan_idx: torch.Tensor,
        spatial_bias_cc: Optional[torch.Tensor],
        spatial_qk_ch: Optional[torch.Tensor],
        grid_channels: Optional[int] = None,
        grid_patches: Optional[int] = None,
    ) -> torch.Tensor:
        attn_bias = None
        spatial_q_add = None
        spatial_k_add = None

        if (spatial_qk_ch is not None) and (chan_idx is not None) and (self.attn.spatial_q_proj is not None):
            q_sp_ch = self.attn.spatial_q_proj(spatial_qk_ch)
            k_sp_ch = self.attn.spatial_k_proj(spatial_qk_ch)
            spatial_q_add = _gather_channel_features(q_sp_ch, chan_idx)
            spatial_k_add = _gather_channel_features(k_sp_ch, chan_idx)
            if padding_mask is not None:
                spatial_q_add = spatial_q_add.masked_fill(padding_mask[..., None], 0.0)
                spatial_k_add = spatial_k_add.masked_fill(padding_mask[..., None], 0.0)

        if self.use_spatial_bias and (spatial_bias_cc is not None) and (chan_idx is not None):
            c = chan_idx
            B, _ = c.shape
            b = torch.arange(B, device=c.device)[:, None, None]
            attn_bias = spatial_bias_cc[b, c[:, :, None], c[:, None, :]]

        y = self.attn(
            self.norm1(x),
            padding_mask=padding_mask,
            rope_pos=rope_pos,
            attn_bias=attn_bias,
            spatial_q_add=spatial_q_add,
            spatial_k_add=spatial_k_add,
        )
        x = x + self.dropout(y)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class DividedSpatiotemporalBlock(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        D = cfg.d_model
        self.norm_t = nn.LayerNorm(D, eps=1e-6)
        self.attn_t = MultiheadSelfAttentionRoPE(
            d_model=D,
            n_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_theta=cfg.rope_theta,
            rotary_pct=cfg.rotary_pct,
            max_seq_len=cfg.max_tokens,
        )

        self.norm_s = nn.LayerNorm(D, eps=1e-6)
        # In the fixed training recipe we use channel-wise spatial bias for divided
        # spatial attention whenever spatial_bias_type is enabled. In that case the
        # additive spatial Q/K path below would be unreachable (the forward path
        # prefers bias over q/k augmentation), which creates permanently-unused
        # parameters under DDP. We therefore only instantiate the divided-block
        # spatial Q/K projections when no spatial bias is active.
        use_divided_spatial_qk = (
            (str(cfg.spatial_qk_type).lower() == "legendre_anchor")
            and (str(cfg.spatial_bias_type).lower() in ("none", "off", "disable", "disabled"))
        )
        self.attn_s = MultiheadSelfAttentionRoPE(
            d_model=D,
            n_heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_theta=cfg.rope_theta,
            rotary_pct=0.0,
            spatial_qk_dim=cfg.spatial_qk_feat_dim,
            spatial_qk_scale=cfg.spatial_qk_scale,
            use_spatial_qk=use_divided_spatial_qk,
            max_seq_len=cfg.max_tokens,
        )

        self.norm_m = nn.LayerNorm(D, eps=1e-6)
        self.mlp = SwiGLU(D, cfg.mlp_ratio, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)
        self._batch_rows_cache = {}
        self._temporal_rope_cache = {}

    @staticmethod
    def _device_key(device: torch.device):
        return (device.type, -1 if device.index is None else int(device.index))

    def _get_batch_rows(self, batch_size: int, device: torch.device) -> torch.Tensor:
        key = (self._device_key(device), int(batch_size))
        rows = self._batch_rows_cache.get(key)
        if rows is None:
            rows = torch.arange(batch_size, device=device, dtype=torch.long)[:, None]
            self._batch_rows_cache[key] = rows
        return rows

    def _get_temporal_rope(self, P: int, device: torch.device) -> torch.Tensor:
        key = (self._device_key(device), int(P))
        rope = self._temporal_rope_cache.get(key)
        if rope is None:
            rope = torch.arange(P, device=device, dtype=torch.long)
            self._temporal_rope_cache[key] = rope
        return rope

    def _scatter_to_grid(self, x: torch.Tensor, pad: torch.Tensor, c_idx: torch.Tensor, t_idx: torch.Tensor, C: int, P: int, b_rows: torch.Tensor):
        B, L, D = x.shape
        grid = x.new_zeros((B, C, P + 1, D))
        grid_pad = torch.ones((B, C, P + 1), dtype=torch.bool, device=x.device)
        b = b_rows.expand(B, L)
        trash = torch.full_like(t_idx, P)
        t_safe = torch.where(pad, trash, t_idx)
        grid[b, c_idx, t_safe] = x
        grid_pad[b, c_idx, t_safe] = pad
        return grid[:, :, :P, :], grid_pad[:, :, :P]

    def _gather_from_grid(self, grid: torch.Tensor, pad: torch.Tensor, c: torch.Tensor, t: torch.Tensor, b_rows: torch.Tensor) -> torch.Tensor:
        B, L = c.shape
        out = grid[b_rows.expand(B, L), c, t]
        return out.masked_fill(pad[..., None], 0.0)

    def _temporal_from_grid(self, grid: torch.Tensor, grid_pad: torch.Tensor, P: int) -> torch.Tensor:
        B, C, P2, D = grid.shape
        assert P2 == P
        x_t = grid.reshape(B * C, P, D)
        pad_t = grid_pad.reshape(B * C, P)
        all_pad = pad_t.all(dim=1)
        pad_t_safe = pad_t.clone()
        pad_t_safe[:, 0] = pad_t_safe[:, 0] & (~all_pad)

        rope = self._get_temporal_rope(P, grid.device)
        y_t = self.attn_t(x_t, padding_mask=pad_t_safe, rope_pos=rope, attn_bias=None)
        y_t = y_t.masked_fill(all_pad[:, None, None], 0.0)
        return y_t.reshape(B, C, P, D)

    def _spatial_from_grid(self, grid: torch.Tensor, grid_pad: torch.Tensor, spatial_bias_cc: Optional[torch.Tensor], spatial_qk_ch: Optional[torch.Tensor], P: int) -> torch.Tensor:
        B, C, P2, D = grid.shape
        assert P2 == P

        grid_tp = grid.permute(0, 2, 1, 3).contiguous()
        pad_tp = grid_pad.permute(0, 2, 1).contiguous()
        x_s = grid_tp.reshape(B * P, C, D)
        pad_s = pad_tp.reshape(B * P, C)

        all_pad = pad_s.all(dim=1)
        pad_s_safe = pad_s.clone()
        pad_s_safe[:, 0] = pad_s_safe[:, 0] & (~all_pad)

        bias = None
        spatial_q_add = None
        spatial_k_add = None
        if spatial_bias_cc is not None:
            bias = spatial_bias_cc[:, None, :, :].expand(-1, P, -1, -1).reshape(B * P, C, C)
        elif (spatial_qk_ch is not None) and (self.attn_s.spatial_q_proj is not None):
            q_sp_ch = self.attn_s.spatial_q_proj(spatial_qk_ch)
            k_sp_ch = self.attn_s.spatial_k_proj(spatial_qk_ch)
            spatial_q_add = q_sp_ch[:, None, :, :].expand(-1, P, -1, -1).reshape(B * P, C, D)
            spatial_k_add = k_sp_ch[:, None, :, :].expand(-1, P, -1, -1).reshape(B * P, C, D)

        y_s = self.attn_s(
            x_s,
            padding_mask=pad_s_safe,
            rope_pos=None,
            attn_bias=bias,
            spatial_q_add=spatial_q_add,
            spatial_k_add=spatial_k_add,
        )
        y_s = y_s.masked_fill(all_pad[:, None, None], 0.0)
        return y_s.reshape(B, P, C, D).permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        rope_pos: torch.Tensor,
        chan_idx: Optional[torch.Tensor],
        spatial_bias_cc: Optional[torch.Tensor],
        spatial_qk_ch: Optional[torch.Tensor],
        grid_channels: Optional[int] = None,
        grid_patches: Optional[int] = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        if chan_idx is None:
            raise ValueError("DividedSpatiotemporalBlock requires chan_idx")

        b_rows = self._get_batch_rows(int(x.shape[0]), x.device)

        if grid_channels is not None:
            C = int(grid_channels)
        elif spatial_bias_cc is not None:
            C = int(spatial_bias_cc.shape[1])
        elif spatial_qk_ch is not None:
            C = int(spatial_qk_ch.shape[1])
        else:
            C = int(chan_idx.max().item()) + 1

        P = int(grid_patches) if grid_patches is not None else int(rope_pos.max().item()) + 1
        grid, grid_pad = self._scatter_to_grid(x, padding_mask, chan_idx, rope_pos, C=C, P=P, b_rows=b_rows)

        grid_normed = self.norm_t(grid).masked_fill(grid_pad[..., None], 0.0)
        grid = grid + self.dropout(self._temporal_from_grid(grid_normed, grid_pad, P=P))

        grid_normed = self.norm_s(grid).masked_fill(grid_pad[..., None], 0.0)
        grid = grid + self.dropout(self._spatial_from_grid(grid_normed, grid_pad, spatial_bias_cc=spatial_bias_cc, spatial_qk_ch=spatial_qk_ch, P=P))

        grid_normed = self.norm_m(grid).masked_fill(grid_pad[..., None], 0.0)
        grid = grid + self.dropout(self.mlp(grid_normed))
        return self._gather_from_grid(grid, padding_mask, chan_idx, rope_pos, b_rows=b_rows)


class EEGEncoder(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_samples = int(round(cfg.sample_rate * cfg.patch_seconds))
        hop_sec = float(cfg.patch_hop_seconds)
        self.hop_samples = max(1, int(round(cfg.sample_rate * hop_sec)))

        self.time_embed = TimePatchEmbed(cfg)
        self.coord_embed = CoordMLPEmbedding(
            d_model=cfg.d_model,
            coord_jitter_std=cfg.coord_jitter_std,
            coord_jitter_prob=cfg.coord_jitter_prob,
            w_init=cfg.coord_w_init,
        )

        self.spatial_qk_feat = None
        if str(cfg.spatial_qk_type).lower() == "legendre_anchor":
            self.spatial_qk_feat = LegendreAnchorFeatures(cfg)

        self.spatial_bias = SpatialBias(cfg)

        arch = str(cfg.encoder_arch).lower()
        full_every = int(cfg.full_attn_every)
        blocks = []
        if arch == "full":
            blocks = [FullAttentionBlock(cfg) for _ in range(cfg.n_layers)]
        elif arch == "divided":
            blocks = [DividedSpatiotemporalBlock(cfg) for _ in range(cfg.n_layers)]
        elif arch == "hybrid":
            for i in range(cfg.n_layers):
                is_full = (full_every > 0) and ((i + 1) % full_every == 0)
                blocks.append(FullAttentionBlock(cfg) if is_full else DividedSpatiotemporalBlock(cfg))
        else:
            raise ValueError(f"Unknown encoder_arch: {arch}")
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(cfg.d_model, eps=1e-6)

    def extract_patches_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.unfold(dimension=-1, size=self.patch_samples, step=self.hop_samples)

    @staticmethod
    def _safe_gather_channel(x: torch.Tensor, c_idx: torch.Tensor) -> torch.Tensor:
        B, C, D = x.shape
        B2, L = c_idx.shape
        assert B == B2
        idx = c_idx[..., None].expand(B, L, D)
        return x.gather(dim=1, index=idx)

    def embed_from_indices(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        c_idx: torch.Tensor,
        t_idx: torch.Tensor,
        pad: torch.Tensor,
        coord_ch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        B, _, _ = x.shape
        B2, L = c_idx.shape
        assert B == B2
        c_safe = c_idx.clamp(min=0)
        t_safe = t_idx.clamp(min=0)

        if coord_ch is None:
            coord_ch = self.coord_embed(coords)
        coord_tok = self._safe_gather_channel(coord_ch, c_safe)
        coord_tok = coord_tok.masked_fill(pad[..., None], 0.0)

        patches_view = self.extract_patches_view(x)
        b_idx = torch.arange(B, device=device)[:, None].expand(B, L)
        patches = patches_view[b_idx, c_safe, t_safe]
        patches = patches.masked_fill(pad[..., None], 0.0)

        tok = self.time_embed.forward_packed(patches) + coord_tok
        tok = tok.masked_fill(pad[..., None], 0.0)
        return tok, pad, t_safe, c_safe

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        rope_pos: torch.Tensor,
        chan_idx: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        grid_patches: Optional[int] = None,
    ) -> torch.Tensor:
        spatial_bias_cc = None
        spatial_qk_ch = None
        grid_channels = None

        if coords is not None:
            grid_channels = int(coords.shape[1])
            spatial_bias_cc = self.spatial_bias(coords)
            if spatial_bias_cc is not None and (spatial_bias_cc.device != tokens.device or spatial_bias_cc.dtype != tokens.dtype):
                spatial_bias_cc = spatial_bias_cc.to(dtype=tokens.dtype, device=tokens.device)

            if self.spatial_qk_feat is not None:
                spatial_qk_ch = self.spatial_qk_feat(coords)
                if spatial_qk_ch.device != tokens.device or spatial_qk_ch.dtype != tokens.dtype:
                    spatial_qk_ch = spatial_qk_ch.to(dtype=tokens.dtype, device=tokens.device)

        if grid_patches is None and rope_pos.numel() > 0:
            grid_patches = int(rope_pos.max().item()) + 1

        x = tokens
        for blk in self.blocks:
            x = blk(
                x,
                padding_mask=padding_mask,
                rope_pos=rope_pos,
                chan_idx=chan_idx,
                spatial_bias_cc=spatial_bias_cc,
                spatial_qk_ch=spatial_qk_ch,
                grid_channels=grid_channels,
                grid_patches=grid_patches,
            )
        x = self.norm(x)
        return x

    def save_pretrained(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg.to_dict(), f, indent=2, ensure_ascii=False)
        torch.save(self.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))

    @staticmethod
    def from_pretrained(path: str, map_location: str = "cpu") -> "EEGEncoder":
        cfg = EEGModelConfig.from_json(os.path.join(path, "config.json"))
        model = EEGEncoder(cfg)
        sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=map_location, weights_only=True)
        model.load_state_dict(sd, strict=True)
        return model


class CrossAttnPredictorBlock(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        d_model = cfg.d_model
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.xattn = CrossAttentionRoPE(
            d_model=d_model,
            n_heads=cfg.predictor_n_heads,
            attn_dropout=cfg.attn_dropout,
            rope_theta=cfg.rope_theta,
            rotary_pct=cfg.rotary_pct,
            max_seq_len=cfg.max_tokens,
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = SwiGLU(d_model, cfg.predictor_mlp_ratio, cfg.dropout)

    def forward(self, q: torch.Tensor, ctx: torch.Tensor, ctx_pad: torch.Tensor, rope_q: torch.Tensor, rope_ctx: torch.Tensor) -> torch.Tensor:
        q = q + self.drop(self.xattn(self.norm1(q), ctx, ctx_pad, rope_pos_q=rope_q, rope_pos_k=rope_ctx))
        q = q + self.drop(self.mlp(self.norm2(q)))
        return q


class CrossAttentionPredictor(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.query_token = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.query_token, std=float(cfg.query_token_init_std))
        self.blocks = nn.ModuleList([CrossAttnPredictorBlock(cfg) for _ in range(int(cfg.predictor_layers))])
        self.norm = nn.LayerNorm(cfg.d_model, eps=1e-6)

    def forward(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        rope_ctx: torch.Tensor,
        tgt_coord_emb: torch.Tensor,
        tgt_pad: torch.Tensor,
        rope_tgt: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query_token[None, None, :].to(tgt_coord_emb.dtype) + tgt_coord_emb
        q = q.masked_fill(tgt_pad[..., None], 0.0)
        for blk in self.blocks:
            q = blk(q, ctx, ctx_pad, rope_q=rope_tgt, rope_ctx=rope_ctx)
        q = self.norm(q)
        return q.masked_fill(tgt_pad[..., None], 0.0)
