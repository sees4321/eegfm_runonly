
from __future__ import annotations

import torch


@torch.no_grad()
def apply_student_augmentations(
    x: torch.Tensor,
    gain_min: float,
    gain_max: float,
    channel_gain_std: float,
    noise_std_min: float,
    noise_std_max: float,
    channel_drop_prob: float,
) -> torch.Tensor:
    """Time-alignment preserving student augmentations for EEG JEPA."""
    if x.numel() == 0:
        return x

    device = x.device
    B, C, _ = x.shape
    x_fp32 = x.to(torch.float32)

    if (gain_min != 1.0) or (gain_max != 1.0):
        g = torch.empty((B, 1, 1), device=device).uniform_(gain_min, gain_max)
        x_fp32 = x_fp32 * g

    if channel_gain_std and channel_gain_std > 0:
        cg = torch.randn((B, C, 1), device=device) * float(channel_gain_std) + 1.0
        cg = torch.clamp(cg, 0.5, 2.0)
        x_fp32 = x_fp32 * cg

    if channel_drop_prob and channel_drop_prob > 0:
        k = max(0, min(C - 1, int(round(C * float(channel_drop_prob)))))
        if k > 0:
            drop = torch.zeros((B, C), device=device, dtype=torch.bool)
            for b in range(B):
                idx = torch.randperm(C, device=device)[:k]
                drop[b, idx] = True
            x_fp32 = x_fp32.masked_fill(drop[:, :, None], 0.0)

    if noise_std_max and noise_std_max > 0:
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=(1, 2), keepdim=True) + 1e-8)
        ns = torch.empty((B, 1, 1), device=device).uniform_(float(noise_std_min), float(noise_std_max))
        noise = torch.randn_like(x_fp32) * (rms * ns)
        x_fp32 = x_fp32 + noise

    return x_fp32.to(x.dtype)
