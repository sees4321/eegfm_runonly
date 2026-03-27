
from __future__ import annotations

from typing import Tuple

import torch


@torch.no_grad()
def _random_nonneg_composition_cpu(total: int, parts: int) -> torch.Tensor:
    out = torch.zeros((parts,), dtype=torch.long)
    if total <= 0 or parts <= 0:
        return out
    idx = torch.randint(0, parts, (total,), dtype=torch.long)
    out.scatter_add_(0, idx, torch.ones((total,), dtype=torch.long))
    return out


@torch.no_grad()
def dilate_time_mask(mask: torch.Tensor, dilation: int) -> torch.Tensor:
    if dilation <= 0:
        return mask
    out = mask.clone()
    for d in range(1, dilation + 1):
        out[..., d:] |= mask[..., :-d]
        out[..., :-d] |= mask[..., d:]
    return out


@torch.no_grad()
def sample_time_mask_style3_same_shape_cpu(
    B: int,
    C: int,
    P: int,
    ratio_min: float,
    ratio_max: float,
) -> torch.Tensor:
    """
    Current run-only path:
      - exact same (C, P) inside a worker batch
      - style-3 temporal keep-block mask
      - CPU only
    """
    mask = torch.zeros((B, C, P), dtype=torch.bool)
    if B <= 0 or C <= 0 or P <= 0:
        return mask

    frac = torch.empty((B,), dtype=torch.float32).uniform_(ratio_min, ratio_max)
    num_masked = torch.round(frac * P).to(torch.long).clamp_(0, P)
    num_kept = P - num_masked
    num_blocks = max(1, P // 10)

    for b in range(B):
        nk = int(num_kept[b])
        nm = int(num_masked[b])

        if nk <= 0:
            mask[b].fill_(True)
            continue
        if nm <= 0:
            continue

        actual_blocks = min(num_blocks, nk)
        base = nk // actual_blocks
        rem = nk - base * actual_blocks

        kept_lengths = torch.full((actual_blocks,), base, dtype=torch.long)
        kept_lengths[-1] += rem
        gaps = _random_nonneg_composition_cpu(nm, actual_blocks + 1)

        row = torch.ones((P,), dtype=torch.bool)
        pos = int(gaps[0])
        for i in range(actual_blocks):
            li = int(kept_lengths[i])
            row[pos:pos + li] = False
            pos += li + int(gaps[i + 1])

        mask[b] = row.unsqueeze(0).expand(C, P)

    return mask


@torch.no_grad()
def sample_spatial_block_mask_same_shape_cpu(
    coords: torch.Tensor,   # (B, C, 3), CPU
    P_t: int,
    ratio_min: float,
    ratio_max: float,
) -> torch.Tensor:
    B, C, _ = coords.shape
    base = torch.zeros((B, C), dtype=torch.bool)
    if B <= 0 or C <= 0 or P_t <= 0:
        return base.unsqueeze(-1).expand(B, C, max(P_t, 1))

    frac = torch.empty((B,), dtype=torch.float32).uniform_(ratio_min, ratio_max)
    k = torch.round(frac * C).to(torch.long).clamp_(1, C)
    centers = torch.randint(0, C, (B,), dtype=torch.long)

    for b in range(B):
        center = coords[b, centers[b]]
        d = torch.sum((coords[b] - center[None, :]) ** 2, dim=-1)
        nn = torch.topk(d, int(k[b]), largest=False).indices
        base[b, nn] = True

    return base.unsqueeze(-1).expand(B, C, P_t)


@torch.no_grad()
def sample_jepa_target_mask_same_shape_style3_cpu(
    coords: torch.Tensor,
    P_t: int,
    mask_time_prob: float,
    mask_spatial_prob: float,
    time_ratio_range: Tuple[float, float],
    spatial_ratio_range: Tuple[float, float],
    dilate_time: int = 0,
) -> torch.Tensor:
    assert coords.device.type == "cpu"
    B, C, _ = coords.shape
    target = torch.zeros((B, C, P_t), dtype=torch.bool)
    if B <= 0 or C <= 0 or P_t <= 0:
        return target

    use_time = torch.rand((B,), dtype=torch.float32) < float(mask_time_prob)
    use_spat = torch.rand((B,), dtype=torch.float32) < float(mask_spatial_prob)
    none = ~(use_time | use_spat)
    use_time[none] = True

    if bool(use_time.any()):
        tmask = sample_time_mask_style3_same_shape_cpu(
            B=B,
            C=C,
            P=P_t,
            ratio_min=float(time_ratio_range[0]),
            ratio_max=float(time_ratio_range[1]),
        )
        target |= tmask & use_time[:, None, None]

    if bool(use_spat.any()):
        smask = sample_spatial_block_mask_same_shape_cpu(
            coords=coords,
            P_t=P_t,
            ratio_min=float(spatial_ratio_range[0]),
            ratio_max=float(spatial_ratio_range[1]),
        )
        target |= smask & use_spat[:, None, None]

    if int(dilate_time) > 0:
        target = dilate_time_mask(target, dilation=int(dilate_time))
    return target


@torch.no_grad()
def mask_to_packed_indices_cpu(mask: torch.Tensor):
    """
    mask: (B, C, P) bool on CPU
    returns: c_idx, t_idx, pad on CPU
    order: time-major (t, c)
    """
    assert mask.device.type == "cpu"
    B, C, P = mask.shape
    mask_tc = mask.permute(0, 2, 1).reshape(B, P * C)
    lengths = mask_tc.sum(dim=1, dtype=torch.long)
    Lmax = int(lengths.max()) if lengths.numel() > 0 else 0
    if Lmax <= 0:
        Lmax = 1

    c_idx = torch.zeros((B, Lmax), dtype=torch.long)
    t_idx = torch.zeros((B, Lmax), dtype=torch.long)
    pad = torch.ones((B, Lmax), dtype=torch.bool)

    nz = mask_tc.nonzero(as_tuple=False)
    if nz.numel() == 0:
        return c_idx, t_idx, pad

    b = nz[:, 0]
    tc = nz[:, 1]
    starts = torch.cumsum(lengths, dim=0) - lengths
    pos = torch.arange(nz.shape[0], dtype=torch.long) - torch.repeat_interleave(starts, lengths)

    t_idx[b, pos] = torch.div(tc, C, rounding_mode="floor")
    c_idx[b, pos] = tc.remainder(C)
    pad[b, pos] = False
    return c_idx, t_idx, pad
