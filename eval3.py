
"""
Sequential LP-initialized cumulative fine-tuning for EEG foundation model checkpoints using cached .npy arrays (memmap).

This script changes the original per-task evaluation flow into a cumulative training flow.

Workflow
--------
- Load one cumulative trainable encoder once from --ckpt (model 1).
- For each task in order:
  1) Load a fresh frozen encoder from the same --ckpt (model 2).
  2) Extract frozen features with model 2 and search for the best LP head on that task.
  3) Copy the best LP head into a classifier built on top of model 1.
  4) Fine-tune model 1 on the task.
  5) Restore the best validation checkpoint for that task and save the cumulative encoder checkpoint.

Notes
-----
- The cumulative encoder is not reset between tasks.
- LP search is always performed on the frozen original pretrained checkpoint.
- LoRA and L2-SP options were intentionally removed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import inspect
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

# Expected: EEGEncoder.from_pretrained(<dir with config.json + pytorch_model.bin>)
from .model import EEGEncoder

try:
    import wandb
except Exception:
    wandb = None


# -------------------------
# utils / dataclasses
# -------------------------
KNOWN_TASK_ALIASES = {
    "physionetmi": "mi",
    "physionet_mi": "mi",
}


@dataclass(frozen=True)
class SplitInfo:
    val_source: str
    label_values: Tuple[int, ...]


@dataclass(frozen=True)
class FeatureNormStats:
    mode: str
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None


@dataclass
class Metrics:
    acc: float
    f1w: float
    kappa: float


@dataclass
class LPResult:
    best_epoch: int
    best_val_acc: float
    test_acc_at_best: float
    best_val_f1w: float
    test_f1w_at_best: float
    test_kappa_at_best: float
    lr: float
    train_acc_at_best: float
    train_acc_last: float
    last_epoch: int
    state_dict: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class FTResult:
    best_epoch: int
    best_val_acc: float
    test_acc_at_best: float
    best_val_f1w: float
    test_f1w_at_best: float
    test_kappa_at_best: float
    backbone_lr: float
    head_lr: float
    train_acc_at_best: float
    train_acc_last: float
    last_epoch: int
    train_scope: str
    unfreeze_last_k: int
    trainable_block_indices: Tuple[int, ...]
    encoder_trainable_params: int
    encoder_total_params: int
    layer_decay: float
    encoder_state_dict: Dict[str, torch.Tensor]
    head_state_dict: Dict[str, torch.Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    group.add_argument(
        f"--no_{name}",
        f"--no-{name}",
        dest=name,
        action="store_false",
        help=f"Disable {name.replace('_', ' ')}.",
    )
    parser.set_defaults(**{name: default})


def feature_dim_from_pool(d_model: int, pool: str) -> int:
    pool = str(pool).lower().strip()
    if pool == "mean":
        return int(d_model)
    if pool in {"mean_std", "tc_mean_std", "ct_mean_std"}:
        return 2 * int(d_model)
    if pool == "tc_ct_mean_std":
        return 4 * int(d_model)
    raise ValueError(f"Unsupported pool={pool}")


def normalize_task_name(name: str) -> str:
    name = str(name).strip()
    base = Path(name).name
    if base.lower().endswith("_npy"):
        base = base[:-4]
    base = base.lower().replace("-", "_").replace(" ", "_")
    return KNOWN_TASK_ALIASES.get(base, base)


def split_dir_has_required_files(split_dir: Path) -> bool:
    return all((split_dir / fname).is_file() for fname in ("eeg.npy", "coords.npy", "label.npy"))


def is_valid_task_root(path: Path) -> bool:
    return (
        path.is_dir()
        and path.name.lower().endswith("_npy")
        and split_dir_has_required_files(path / "train")
        and split_dir_has_required_files(path / "test")
    )


def discover_npy_task_roots(base_dir: str) -> Dict[str, str]:
    base = Path(base_dir).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"task_root not found: {base}")

    roots: Dict[str, str] = {}
    candidates: List[Path] = []
    if is_valid_task_root(base):
        candidates.append(base)
    candidates.extend(sorted(p for p in base.rglob("*_npy") if p.is_dir()))

    for p in candidates:
        if not is_valid_task_root(p):
            continue
        task = normalize_task_name(p.name)
        prev = roots.get(task)
        if prev is not None and Path(prev).expanduser().resolve() != p.resolve():
            raise ValueError(f"Duplicate discovered task name '{task}' for roots: {prev} and {p}")
        roots[task] = str(p)
    return roots


def resolve_task_specs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    roots: Dict[str, str] = {}

    def _register(task_name: str, root_path: str, *, allow_override: bool = False) -> None:
        key = normalize_task_name(task_name)
        path = str(Path(root_path).expanduser())
        prev = roots.get(key)
        if prev is None:
            roots[key] = path
            return
        same = Path(prev).expanduser().resolve() == Path(path).expanduser().resolve()
        if same:
            return
        if allow_override:
            roots[key] = path
            return
        raise ValueError(f"Duplicate task root for task='{key}': {prev} vs {path}")

    task_root = getattr(args, "task_root", "")
    if task_root:
        for task, root in discover_npy_task_roots(task_root).items():
            _register(task, root, allow_override=False)

    explicit_roots = {
        "tuab": getattr(args, "tuab_root", ""),
        "isruc": getattr(args, "isruc_root", ""),
        "mi": getattr(args, "mi_root", ""),
    }
    for task, root in explicit_roots.items():
        if root:
            _register(task, root, allow_override=True)

    if not roots:
        raise ValueError("No task roots found. Provide --task_root and/or explicit per-task roots.")

    requested_tasks = getattr(args, "tasks", None)
    if requested_tasks:
        ordered: List[Tuple[str, str]] = []
        seen = set()
        for raw_task in requested_tasks:
            task = normalize_task_name(raw_task)
            if task not in roots:
                available = ", ".join(sorted(roots.keys()))
                raise ValueError(f"Unknown task='{raw_task}' (normalized='{task}'). Available tasks: {available}")
            if task in seen:
                continue
            ordered.append((task, roots[task]))
            seen.add(task)
        return ordered

    return [(task, roots[task]) for task in sorted(roots.keys())]


def compute_num_patches(T: int, patch_samples: int, hop_samples: int) -> int:
    if T < patch_samples:
        return 0
    return int((T - patch_samples) // hop_samples + 1)


def maybe_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cpu_clone_state_dict(module_or_state_dict) -> Dict[str, torch.Tensor]:
    if isinstance(module_or_state_dict, dict):
        state_dict = module_or_state_dict
    else:
        state_dict = module_or_state_dict.state_dict()
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def is_better_candidate(
    cand_val_acc: float,
    cand_val_f1w: float,
    best_val_acc: float,
    best_val_f1w: float,
) -> bool:
    if cand_val_acc > best_val_acc + 1e-6:
        return True
    if abs(cand_val_acc - best_val_acc) <= 1e-6 and cand_val_f1w > best_val_f1w + 1e-6:
        return True
    return False


def serialize_feature_norm_stats(stats: FeatureNormStats) -> Dict[str, object]:
    out: Dict[str, object] = {"mode": str(stats.mode)}
    if stats.mean is not None:
        out["mean"] = stats.mean.detach().cpu()
    if stats.std is not None:
        out["std"] = stats.std.detach().cpu()
    return out


# -------------------------
# Dataset helpers
# -------------------------
class NpySplitDataset(Dataset):
    def __init__(self, split_dir: str):
        split_dir = str(split_dir)
        self.split_dir = split_dir
        eeg_path = os.path.join(split_dir, "eeg.npy")
        coords_path = os.path.join(split_dir, "coords.npy")
        label_path = os.path.join(split_dir, "label.npy")

        if not os.path.exists(eeg_path):
            raise FileNotFoundError(eeg_path)
        if not os.path.exists(coords_path):
            raise FileNotFoundError(coords_path)
        if not os.path.exists(label_path):
            raise FileNotFoundError(label_path)

        self.eeg = np.load(eeg_path, mmap_mode="r")
        self.coords = np.load(coords_path, mmap_mode="r")
        self.y = np.load(label_path, mmap_mode="r")

        if self.eeg.shape[0] != self.coords.shape[0] or self.eeg.shape[0] != self.y.shape[0]:
            raise ValueError(f"N mismatch: eeg={self.eeg.shape} coords={self.coords.shape} y={self.y.shape}")

        if self.eeg.ndim != 3:
            raise ValueError(f"eeg must be (N,C,T), got {self.eeg.shape}")
        if self.coords.ndim != 3 or self.coords.shape[-1] != 3:
            raise ValueError(f"coords must be (N,C,3), got {self.coords.shape}")

    def __len__(self) -> int:
        return int(self.eeg.shape[0])

    def __getitem__(self, idx: int):
        return self.eeg[idx], self.coords[idx], int(self.y[idx])


class RemappedLabelDataset(Dataset):
    def __init__(self, base: Dataset, label_map: Dict[int, int]):
        self.base = base
        self.label_map = {int(k): int(v) for k, v in label_map.items()}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        eeg, coord, y = self.base[idx]
        return eeg, coord, self.label_map[int(y)]


def infer_label_mapping(*datasets: NpySplitDataset) -> Tuple[Dict[int, int], Tuple[int, ...]]:
    labels = set()
    for ds in datasets:
        if ds is None:
            continue
        y = np.asarray(ds.y, dtype=np.int64).reshape(-1)
        labels.update(int(v) for v in np.unique(y))
    if not labels:
        raise ValueError("Could not infer labels from datasets.")
    label_values = tuple(sorted(labels))
    label_map = {orig: idx for idx, orig in enumerate(label_values)}
    return label_map, label_values


def stratified_split_indices(y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    assert y.ndim == 1
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx: List[int] = []
    val_idx: List[int] = []
    all_idx = np.arange(len(y))

    for c in classes:
        idx_c = all_idx[y == c]
        rng.shuffle(idx_c)
        if len(idx_c) <= 1:
            train_idx.extend(idx_c.tolist())
            continue
        n_val = int(round(len(idx_c) * float(val_ratio)))
        n_val = min(max(n_val, 1), len(idx_c) - 1)
        val_idx.extend(idx_c[:n_val].tolist())
        train_idx.extend(idx_c[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def load_task_splits(task: str, root: str, seed: int, missing_val_ratio: float) -> Tuple[Dataset, Dataset, Dataset, int, SplitInfo]:
    root_path = Path(root).expanduser()
    train_dir = root_path / "train"
    test_dir = root_path / "test"
    val_dir = root_path / "val"

    if not split_dir_has_required_files(train_dir):
        raise FileNotFoundError(f"Missing required train split files under: {train_dir}")
    if not split_dir_has_required_files(test_dir):
        raise FileNotFoundError(f"Missing required test split files under: {test_dir}")

    train_base = NpySplitDataset(str(train_dir))
    test_base = NpySplitDataset(str(test_dir))

    if val_dir.exists():
        if not split_dir_has_required_files(val_dir):
            raise FileNotFoundError(f"val/ exists but is incomplete under: {val_dir}")
        val_base = NpySplitDataset(str(val_dir))
        train_raw: Dataset = train_base
        val_raw: Dataset = val_base
        val_source = "provided"
        label_map, label_values = infer_label_mapping(train_base, val_base, test_base)
    else:
        y_train = np.asarray(train_base.y, dtype=np.int64)
        tr_idx, va_idx = stratified_split_indices(y_train, val_ratio=missing_val_ratio, seed=seed)
        if len(va_idx) == 0:
            raise RuntimeError(
                f"Task '{task}' at {root} has no val/ folder and stratified split produced an empty val set."
            )
        train_raw = Subset(train_base, tr_idx.tolist())
        val_raw = Subset(train_base, va_idx.tolist())
        val_source = f"split_from_train_{missing_val_ratio:.3f}"
        label_map, label_values = infer_label_mapping(train_base, test_base)

    train_ds = RemappedLabelDataset(train_raw, label_map)
    val_ds = RemappedLabelDataset(val_raw, label_map)
    test_ds = RemappedLabelDataset(test_base, label_map)
    info = SplitInfo(val_source=val_source, label_values=label_values)
    return train_ds, val_ds, test_ds, len(label_values), info


def make_collate_eeg(coord_scale: float = 10.0):
    coord_scale = float(coord_scale)

    def _collate(batch):
        eegs, coords, ys = zip(*batch)
        eeg = torch.from_numpy(np.stack(eegs, axis=0))
        coord = torch.from_numpy(np.stack(coords, axis=0))
        y = torch.tensor(ys, dtype=torch.long)

        if eeg.dtype != torch.float16:
            eeg = eeg.to(torch.float16)
        coord = coord.to(torch.float32) * coord_scale
        return eeg, coord, y

    return _collate


# -------------------------
# Encoder call helpers + pooling
# -------------------------
def _filter_supported_kwargs(fn, kwargs: Dict[str, object]) -> Dict[str, object]:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _call_embed_from_indices(encoder: EEGEncoder, **kwargs):
    embed_kwargs = _filter_supported_kwargs(encoder.embed_from_indices, kwargs)
    out = encoder.embed_from_indices(**embed_kwargs)
    if isinstance(out, (list, tuple)):
        if len(out) == 3:
            tok, pad, rope_pos = out
            chan_idx = None
            return tok, pad, rope_pos, chan_idx
        if len(out) == 4:
            tok, pad, rope_pos, chan_idx = out
            return tok, pad, rope_pos, chan_idx
    raise RuntimeError(f"Unexpected embed_from_indices return: type={type(out)}")


def _call_encoder_forward(
    encoder: EEGEncoder,
    tok: torch.Tensor,
    pad: torch.Tensor,
    rope_pos: torch.Tensor,
    coords: torch.Tensor,
    chan_idx: Optional[torch.Tensor],
    grid_patches: Optional[int] = None,
) -> torch.Tensor:
    forward_kwargs = {
        "padding_mask": pad,
        "rope_pos": rope_pos,
        "coords": coords,
    }
    if chan_idx is not None:
        forward_kwargs["chan_idx"] = chan_idx
    if grid_patches is not None:
        forward_kwargs["grid_patches"] = int(grid_patches)

    kwargs = _filter_supported_kwargs(encoder.forward, forward_kwargs)
    return encoder(tok, **kwargs)


def pool_tokens(
    z: torch.Tensor,
    pad: Optional[torch.Tensor],
    pool: str,
    *,
    C: Optional[int] = None,
    P: Optional[int] = None,
) -> torch.Tensor:
    pool = str(pool).lower().strip()
    B, L, D = z.shape

    if pad is None:
        valid = torch.ones((B, L), device=z.device, dtype=torch.bool)
    else:
        valid = ~pad

    def masked_mean_and_std(x: torch.Tensor, v: torch.Tensor, dim: int):
        w = v.to(x.dtype)
        denom = w.sum(dim=dim, keepdim=True).clamp_min(1.0)
        mean = (x * w.unsqueeze(-1)).sum(dim=dim, keepdim=True) / denom
        var = ((x - mean) ** 2 * w.unsqueeze(-1)).sum(dim=dim, keepdim=True) / denom
        std = torch.sqrt(torch.clamp(var, min=0.0) + 1e-6)
        return mean.squeeze(dim), std.squeeze(dim)

    if pool == "mean":
        w = valid.to(z.dtype)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (z * w[..., None]).sum(dim=1) / denom

    if pool == "mean_std":
        w = valid.to(z.dtype)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (z * w[..., None]).sum(dim=1) / denom
        mean2 = (z * z * w[..., None]).sum(dim=1) / denom
        var = torch.clamp(mean2 - mean * mean, min=0.0)
        std = torch.sqrt(var + 1e-6)
        return torch.cat([mean, std], dim=-1)

    if C is None or P is None:
        raise ValueError(f"pool={pool} requires C and P, got C={C}, P={P}")
    if L != int(C) * int(P):
        raise ValueError(f"pool={pool} requires L==C*P. Got L={L}, C={C}, P={P}")

    C = int(C)
    P = int(P)
    z_grid = z.view(B, P, C, D)
    v_grid = valid.view(B, P, C)

    if pool in ("tc_mean_std", "tc_ct_mean_std"):
        v_c = v_grid
        w_c = v_c.to(z.dtype)
        denom_c = w_c.sum(dim=2, keepdim=True).clamp_min(1.0)
        z_t = (z_grid * w_c.unsqueeze(-1)).sum(dim=2) / denom_c
        v_t = v_c.any(dim=2)
        mean_t, std_t = masked_mean_and_std(z_t, v_t, dim=1)
        feat_tc = torch.cat([mean_t, std_t], dim=-1)

    if pool == "tc_mean_std":
        return feat_tc

    if pool in ("ct_mean_std", "tc_ct_mean_std"):
        v_p = v_grid
        w_p = v_p.to(z.dtype)
        denom_p = w_p.sum(dim=1, keepdim=True).clamp_min(1.0)
        z_c = (z_grid * w_p.unsqueeze(-1)).sum(dim=1) / denom_p.squeeze(1).unsqueeze(-1)
        v_c2 = v_p.any(dim=1)
        mean_c, std_c = masked_mean_and_std(z_c, v_c2, dim=1)
        feat_ct = torch.cat([mean_c, std_c], dim=-1)

    if pool == "ct_mean_std":
        return feat_ct

    if pool == "tc_ct_mean_std":
        return torch.cat([feat_tc, feat_ct], dim=-1)

    raise ValueError(f"Unknown pool: {pool}")


@torch.no_grad()
def rescale_small_segments(
    x: torch.Tensor,
    target_amp: float = 1.0,
    quantile: float = 0.90,
    amp_floor: float = 1e-4,
    gain_max: float = 200.0,
    clip: float = 15.0,
) -> torch.Tensor:
    x32 = x.float()
    T = x32.shape[-1]
    k_idx = max(1, int(round(float(quantile) * float(T))))
    robust_amp = x32.abs().kthvalue(k_idx, dim=-1, keepdim=True).values
    gain = (float(target_amp) / robust_amp.clamp_min(float(amp_floor))).clamp(max=float(gain_max))
    x32 = x32 * gain
    if clip and clip > 0:
        x32 = x32.clamp(-clip, clip)
    return x32.to(dtype=x.dtype)


def compute_feature_norm_stats(X_train: torch.Tensor, mode: str) -> FeatureNormStats:
    mode = (mode or "none").lower().strip()
    if mode == "none":
        return FeatureNormStats(mode=mode)
    if mode == "zscore":
        mu = X_train.mean(dim=0, keepdim=True).float().cpu()
        sd = X_train.std(dim=0, keepdim=True).clamp_min(1e-6).float().cpu()
        return FeatureNormStats(mode=mode, mean=mu, std=sd)
    if mode == "l2":
        return FeatureNormStats(mode=mode)
    raise ValueError(f"Unknown feat_norm={mode}")


def apply_feature_norm(x: torch.Tensor, stats: FeatureNormStats) -> torch.Tensor:
    mode = str(stats.mode).lower().strip()
    if mode == "none":
        return x
    if mode == "zscore":
        if stats.mean is None or stats.std is None:
            raise ValueError("zscore normalization requires mean/std statistics.")
        mean = stats.mean.to(device=x.device, dtype=x.dtype)
        std = stats.std.to(device=x.device, dtype=x.dtype).clamp_min(1e-6)
        return (x - mean) / std
    if mode == "l2":
        return F.normalize(x, p=2, dim=-1)
    raise ValueError(f"Unknown feat_norm={mode}")


def normalize_features(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    X_test: torch.Tensor,
    stats: FeatureNormStats,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        apply_feature_norm(X_train, stats),
        apply_feature_norm(X_val, stats),
        apply_feature_norm(X_test, stats),
    )


# -------------------------
# model wrappers
# -------------------------
class EncoderFeaturizer(nn.Module):
    def __init__(
        self,
        encoder: EEGEncoder,
        *,
        pool: str,
        amp: bool,
        apply_rescale: bool,
        rescale_kwargs: Dict[str, float],
    ):
        super().__init__()
        self.encoder = encoder
        self.pool = str(pool)
        self.amp = bool(amp)
        self.apply_rescale = bool(apply_rescale)
        self.rescale_kwargs = dict(rescale_kwargs)
        self.patch_samples = int(getattr(encoder, "patch_samples"))
        self.hop_samples = int(getattr(encoder, "hop_samples"))
        self.feat_dim = int(feature_dim_from_pool(int(encoder.cfg.d_model), self.pool))

    def forward(self, eeg: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        if self.apply_rescale:
            eeg = rescale_small_segments(eeg, **self.rescale_kwargs)

        if eeg.ndim != 3:
            raise ValueError(f"Expected eeg with shape (B,C,T), got {tuple(eeg.shape)}")

        B, C, T = eeg.shape
        P = compute_num_patches(T, self.patch_samples, self.hop_samples)
        if P <= 0:
            raise RuntimeError(
                f"Invalid number of patches P={P} from T={T}, patch_samples={self.patch_samples}, hop_samples={self.hop_samples}"
            )

        L = C * P
        device = eeg.device
        c_idx = torch.arange(C, device=device, dtype=torch.long).repeat(P)[None, :].expand(B, L)
        t_idx = torch.arange(P, device=device, dtype=torch.long).repeat_interleave(C)[None, :].expand(B, L)
        pad = torch.zeros((B, L), dtype=torch.bool, device=device)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(self.amp and device.type == "cuda")):
            coord_ch = None
            if hasattr(self.encoder, "coord_embed"):
                coord_ch = self.encoder.coord_embed(coord)

            tok, pad2, rope, chan = _call_embed_from_indices(
                self.encoder,
                x=eeg,
                coords=coord,
                c_idx=c_idx,
                t_idx=t_idx,
                pad=pad,
                coord_ch=coord_ch,
            )
            z = _call_encoder_forward(
                self.encoder,
                tok=tok,
                pad=pad2,
                rope_pos=rope,
                coords=coord,
                chan_idx=chan,
                grid_patches=P,
            )
            feat = pool_tokens(z, pad2, pool=self.pool, C=C, P=P)

        return feat.float()


class FeatureNormalizer(nn.Module):
    def __init__(self, stats: FeatureNormStats):
        super().__init__()
        self.mode = str(stats.mode).lower().strip()
        if self.mode == "zscore":
            if stats.mean is None or stats.std is None:
                raise ValueError("zscore normalization requires mean/std statistics.")
            self.register_buffer("mean", stats.mean.detach().float().clone())
            self.register_buffer("std", stats.std.detach().float().clone().clamp_min(1e-6))
        else:
            self.register_buffer("mean", torch.empty(0))
            self.register_buffer("std", torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        if self.mode == "zscore":
            return (x - self.mean.to(device=x.device, dtype=x.dtype)) / self.std.to(device=x.device, dtype=x.dtype)
        if self.mode == "l2":
            return F.normalize(x, p=2, dim=-1)
        raise ValueError(f"Unknown feat_norm={self.mode}")


class EEGHead(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int, layers: int = 3):
        super().__init__()
        if layers == 3:
            self.head = nn.Sequential(
                nn.Linear(int(feat_dim), int(feat_dim // 4)),
                nn.ELU(),
                nn.Linear(int(feat_dim // 4), int(feat_dim // 16)),
                nn.ELU(),
                nn.Linear(int(feat_dim // 16), int(n_classes)),
            )
        elif layers == 2:
            self.head = nn.Sequential(
                nn.Linear(int(feat_dim), int(feat_dim // 4)),
                nn.ELU(),
                nn.Linear(int(feat_dim // 4), int(n_classes)),
            )
        else:
            self.head = nn.Linear(int(feat_dim), int(n_classes))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)


class EEGClassifier(nn.Module):
    def __init__(self, feature_model: EncoderFeaturizer, n_classes: int, norm_stats: FeatureNormStats):
        super().__init__()
        self.feature_model = feature_model
        self.feature_norm = FeatureNormalizer(norm_stats)
        self.head = EEGHead(int(feature_model.feat_dim), int(n_classes))

    def forward(self, eeg: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        feat = self.feature_model(eeg, coord)
        feat = self.feature_norm(feat)
        return self.head(feat)


def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


# -------------------------
# metrics / evaluation
# -------------------------
@torch.no_grad()
def _confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int, device: torch.device) -> torch.Tensor:
    y_true = y_true.to(device=device, dtype=torch.long)
    y_pred = y_pred.to(device=device, dtype=torch.long)
    idx = y_true * n_classes + y_pred
    cm = torch.bincount(idx, minlength=n_classes * n_classes)
    return cm.view(n_classes, n_classes)


@torch.no_grad()
def weighted_f1_from_cm(cm: torch.Tensor) -> float:
    cm = cm.to(torch.float32)
    tp = torch.diagonal(cm)
    support = cm.sum(dim=1)
    fp = cm.sum(dim=0) - tp
    fn = support - tp
    denom = (2 * tp + fp + fn).clamp_min(1e-12)
    f1 = (2 * tp) / denom
    w = support.clamp_min(0.0)
    return float((f1 * w).sum().div(w.sum().clamp_min(1e-12)).item())


@torch.no_grad()
def cohen_kappa_from_cm(cm: torch.Tensor) -> float:
    cm = cm.to(torch.float32)
    n = float(cm.sum().item())
    if n <= 0.0:
        return 0.0
    po = float(torch.diagonal(cm).sum().item() / n)
    row = cm.sum(dim=1)
    col = cm.sum(dim=0)
    pe = float((row * col).sum().item() / max(n * n, 1e-12))
    denom = 1.0 - pe
    if abs(denom) < 1e-12:
        return 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    return float((po - pe) / denom)


@torch.no_grad()
def metrics_from_cm(cm: torch.Tensor) -> Metrics:
    cmf = cm.to(torch.float32)
    total = float(cmf.sum().item())
    acc = 0.0 if total <= 0.0 else float(torch.diagonal(cmf).sum().item() / total)
    f1w = weighted_f1_from_cm(cmf)
    kappa = cohen_kappa_from_cm(cmf)
    return Metrics(acc=acc, f1w=f1w, kappa=kappa)


def binary_average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(f"Shape mismatch for AP: y_true={y_true.shape}, y_score={y_score.shape}")

    n_pos = int(y_true.sum())
    if n_pos <= 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    distinct = np.where(np.diff(score_sorted))[0]
    threshold_idx = np.r_[distinct, y_sorted.size - 1]

    tps = tp[threshold_idx].astype(np.float64)
    fps = fp[threshold_idx].astype(np.float64)
    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tps / float(n_pos)

    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += float(r - prev_recall) * float(p)
        prev_recall = float(r)
    return float(ap)


def binary_auroc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(f"Shape mismatch for AUROC: y_true={y_true.shape}, y_score={y_score.shape}")

    n_pos = int(y_true.sum())
    n_neg = int(y_true.shape[0] - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    scores_sorted = y_score[order]

    ranks_sorted = np.empty_like(scores_sorted, dtype=np.float64)
    start = 0
    while start < scores_sorted.shape[0]:
        end = start + 1
        while end < scores_sorted.shape[0] and scores_sorted[end] == scores_sorted[start]:
            end += 1
        avg_rank = 0.5 * ((start + 1) + end)
        ranks_sorted[start:end] = avg_rank
        start = end

    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def evaluate_head(
    head: nn.Module,
    loader: DataLoader,
    n_classes: int,
    device: torch.device,
    *,
    binary_test_metrics: bool = False,
) -> Metrics:
    head.eval()
    if binary_test_metrics and n_classes == 2:
        y_true_parts: List[np.ndarray] = []
        y_score_parts: List[np.ndarray] = []
        correct = 0
        total = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = head(xb)
            pred = logits.argmax(dim=-1)
            scores = (logits[:, 1] - logits[:, 0]).detach().float().cpu().numpy()

            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
            y_true_parts.append(yb.detach().cpu().numpy())
            y_score_parts.append(scores)

        if total == 0:
            return Metrics(acc=0.0, f1w=float("nan"), kappa=float("nan"))

        y_true = np.concatenate(y_true_parts, axis=0)
        y_score = np.concatenate(y_score_parts, axis=0)
        acc = float(correct / total)
        auc_pr = binary_average_precision_score(y_true, y_score)
        auroc = binary_auroc_score(y_true, y_score)
        return Metrics(acc=acc, f1w=auc_pr, kappa=auroc)

    cm = torch.zeros((n_classes, n_classes), dtype=torch.long, device=device)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = head(xb)
        pred = logits.argmax(dim=-1)
        cm += _confusion_matrix(yb, pred, n_classes=n_classes, device=device)
    return metrics_from_cm(cm)


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    n_classes: int,
    device: torch.device,
    *,
    binary_test_metrics: bool = False,
) -> Metrics:
    model.eval()
    if binary_test_metrics and n_classes == 2:
        y_true_parts: List[np.ndarray] = []
        y_score_parts: List[np.ndarray] = []
        correct = 0
        total = 0

        for eeg, coord, yb in loader:
            eeg = eeg.to(device, non_blocking=True)
            coord = coord.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(eeg, coord)
            pred = logits.argmax(dim=-1)
            scores = (logits[:, 1] - logits[:, 0]).detach().float().cpu().numpy()

            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
            y_true_parts.append(yb.detach().cpu().numpy())
            y_score_parts.append(scores)

        if total == 0:
            return Metrics(acc=0.0, f1w=float("nan"), kappa=float("nan"))

        y_true = np.concatenate(y_true_parts, axis=0)
        y_score = np.concatenate(y_score_parts, axis=0)
        acc = float(correct / total)
        auc_pr = binary_average_precision_score(y_true, y_score)
        auroc = binary_auroc_score(y_true, y_score)
        return Metrics(acc=acc, f1w=auc_pr, kappa=auroc)

    cm = torch.zeros((n_classes, n_classes), dtype=torch.long, device=device)
    for eeg, coord, yb in loader:
        eeg = eeg.to(device, non_blocking=True)
        coord = coord.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(eeg, coord)
        pred = logits.argmax(dim=-1)
        cm += _confusion_matrix(yb, pred, n_classes=n_classes, device=device)
    return metrics_from_cm(cm)


# -------------------------
# Feature extraction
# -------------------------
@torch.no_grad()
def extract_features(
    feature_model: nn.Module,
    ds: Dataset,
    device: torch.device,
    feat_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    coord_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feature_model.eval()

    loader_kwargs = dict(
        batch_size=feat_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=make_collate_eeg(coord_scale),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(ds, **loader_kwargs)

    base_model = unwrap_module(feature_model)
    feat_dim = int(getattr(base_model, "feat_dim"))
    N = len(ds)
    X = torch.empty((N, feat_dim), dtype=torch.float32)
    y_all = torch.empty((N,), dtype=torch.long)

    offset = 0
    for eeg, coord, y in loader:
        B = eeg.shape[0]
        eeg = eeg.to(device, non_blocking=True)
        coord = coord.to(device, non_blocking=True)
        feat = feature_model(eeg, coord)
        feat_cpu = feat.detach().float().cpu()
        X[offset : offset + B] = feat_cpu
        y_all[offset : offset + B] = y.cpu()
        offset += B

    assert offset == N
    return X, y_all


# -------------------------
# Linear probe training
# -------------------------
def train_linear_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_classes: int,
    device: torch.device,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    weight_decay: float = 0.0,
    class_weight: Optional[str] = None,
    seed: int = 42,
) -> LPResult:
    set_seed(seed)
    head = EEGHead(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    if class_weight == "balanced":
        with torch.no_grad():
            counts = torch.bincount(y_train.cpu(), minlength=n_classes).float()
            w = (counts.sum() / counts.clamp_min(1.0))
            w = (w / w.mean()).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=w)
    else:
        loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=train_gen,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    best_val = -1.0
    best_epoch = -1
    best_state = None
    best_test_acc = -1.0
    best_val_f1w = -1.0
    best_test_f1w = -1.0
    best_test_kappa = -1.0
    bad = 0

    train_acc_at_best = 0.0
    train_acc_last = 0.0
    last_epoch = 0

    for ep in range(1, epochs + 1):
        head.train()
        correct_tr = 0
        total_tr = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            pred = logits.argmax(dim=-1)
            correct_tr += int((pred == yb).sum().item())
            total_tr += int(yb.numel())
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        train_acc = correct_tr / max(1, total_tr)
        train_acc_last = float(train_acc)
        last_epoch = int(ep)

        with torch.no_grad():
            val_metrics = evaluate_head(head, val_loader, n_classes=n_classes, device=device)

        if is_better_candidate(val_metrics.acc, val_metrics.f1w, best_val, best_val_f1w):
            with torch.no_grad():
                test_metrics = evaluate_head(
                    head,
                    test_loader,
                    n_classes=n_classes,
                    device=device,
                    binary_test_metrics=(n_classes == 2),
                )
            best_val = float(val_metrics.acc)
            best_epoch = int(ep)
            best_test_acc = float(test_metrics.acc)
            best_val_f1w = float(val_metrics.f1w)
            best_test_f1w = float(test_metrics.f1w)
            best_test_kappa = float(test_metrics.kappa)
            best_state = cpu_clone_state_dict(head)
            train_acc_at_best = float(train_acc_last)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        best_state = cpu_clone_state_dict(head)

    head.load_state_dict(best_state)

    return LPResult(
        best_epoch=best_epoch,
        best_val_acc=best_val,
        test_acc_at_best=best_test_acc,
        best_val_f1w=best_val_f1w,
        test_f1w_at_best=best_test_f1w,
        test_kappa_at_best=best_test_kappa,
        lr=float(lr),
        train_acc_at_best=float(train_acc_at_best),
        train_acc_last=float(train_acc_last),
        last_epoch=int(last_epoch),
        state_dict=cpu_clone_state_dict(head),
    )


# -------------------------
# FT helpers
# -------------------------
def _make_eeg_loader(
    ds: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    coord_scale: float,
) -> DataLoader:
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=make_collate_eeg(coord_scale),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        loader_kwargs["generator"] = g
    return DataLoader(ds, **loader_kwargs)


def _set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(requires_grad)


def _count_params(params) -> int:
    return sum(int(p.numel()) for p in params)


def _resolve_ft_unfreeze_last_k(num_blocks: int, requested_k: int) -> int:
    num_blocks = int(num_blocks)
    requested_k = int(requested_k)
    if num_blocks <= 0:
        return 0
    if requested_k > 0:
        return min(requested_k, num_blocks)
    if num_blocks <= 4:
        return max(1, num_blocks // 2)
    return min(num_blocks, max(2, int(math.ceil(0.25 * float(num_blocks)))))


def _build_param_groups_for_module(
    module: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    group_name: str,
) -> List[Dict[str, object]]:
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, param in module.named_parameters(recurse=True):
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: List[Dict[str, object]] = []
    if decay_params:
        groups.append({
            "params": decay_params,
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "group_name": str(group_name),
        })
    if no_decay_params:
        groups.append({
            "params": no_decay_params,
            "lr": float(lr),
            "weight_decay": 0.0,
            "group_name": f"{group_name}/no_decay",
        })
    return groups


def _select_encoder_modules_for_scope(
    encoder: EEGEncoder,
    *,
    train_scope: str,
    unfreeze_last_k: int,
    include_full_prefix_modules: bool,
    include_norm: bool,
) -> Tuple[List[Tuple[str, nn.Module]], Dict[str, object]]:
    train_scope = str(train_scope).lower().strip()
    if train_scope not in {"last_k", "full"}:
        raise ValueError(f"Unsupported ft_train_scope={train_scope}")

    block_list = list(getattr(encoder, "blocks", []))
    n_blocks = len(block_list)
    selected_modules_info: List[Tuple[str, nn.Module]] = []
    trainable_block_indices: List[int] = []

    def _register_module(name: str, module: Optional[nn.Module]) -> None:
        if module is None:
            return
        if _count_params(module.parameters()) <= 0:
            return
        selected_modules_info.append((str(name), module))

    if train_scope == "last_k":
        if n_blocks <= 0:
            raise RuntimeError("ft_train_scope=last_k requires encoder.blocks to exist and be non-empty.")
        effective_k = _resolve_ft_unfreeze_last_k(n_blocks, unfreeze_last_k)
        start_idx = max(0, n_blocks - effective_k)
        for idx in range(start_idx, n_blocks):
            _register_module(f"blocks.{idx}", block_list[idx])
            trainable_block_indices.append(idx)
    else:
        effective_k = n_blocks
        if include_full_prefix_modules:
            for name in ("time_embed", "coord_embed", "spatial_qk_feat", "spatial_bias"):
                _register_module(name, getattr(encoder, name, None))
        for idx, blk in enumerate(block_list):
            _register_module(f"blocks.{idx}", blk)
            trainable_block_indices.append(idx)

    if include_norm:
        _register_module("norm", getattr(encoder, "norm", None))

    if not selected_modules_info:
        raise RuntimeError("No encoder modules were selected for fine-tuning.")

    plan_base = {
        "train_scope": train_scope,
        "unfreeze_last_k": int(effective_k),
        "trainable_block_indices": tuple(int(v) for v in trainable_block_indices),
    }
    return selected_modules_info, plan_base


def build_ft_encoder_param_groups(
    encoder: EEGEncoder,
    *,
    train_scope: str,
    unfreeze_last_k: int,
    backbone_lr: float,
    layer_decay: float,
    weight_decay: float,
) -> Tuple[List[Dict[str, object]], List[nn.Module], Dict[str, object]]:
    if not (0.0 < float(layer_decay) <= 1.0):
        raise ValueError("ft_layer_decay must satisfy 0 < value <= 1.")

    _set_module_requires_grad(encoder, False)
    trainable_modules_info, plan_base = _select_encoder_modules_for_scope(
        encoder,
        train_scope=train_scope,
        unfreeze_last_k=unfreeze_last_k,
        include_full_prefix_modules=(str(train_scope).lower().strip() == "full"),
        include_norm=True,
    )
    for _, module in trainable_modules_info:
        _set_module_requires_grad(module, True)

    param_groups: List[Dict[str, object]] = []
    n_stages = len(trainable_modules_info)
    for stage_idx, (name, module) in enumerate(trainable_modules_info):
        depth_from_top = n_stages - stage_idx - 1
        stage_lr = float(backbone_lr) * (float(layer_decay) ** depth_from_top)
        param_groups.extend(
            _build_param_groups_for_module(
                module,
                lr=stage_lr,
                weight_decay=weight_decay,
                group_name=name,
            )
        )

    encoder_total_params = _count_params(encoder.parameters())
    encoder_trainable_params = _count_params(p for p in encoder.parameters() if p.requires_grad)
    plan = {
        **plan_base,
        "encoder_trainable_params": int(encoder_trainable_params),
        "encoder_total_params": int(encoder_total_params),
        "layer_decay": float(layer_decay),
    }
    return param_groups, [m for _, m in trainable_modules_info], plan


def _set_task_ft_train_mode(model: EEGClassifier, trainable_encoder_modules: Sequence[nn.Module]) -> None:
    model.head.train()
    encoder = model.feature_model.encoder
    encoder.eval()
    for module in trainable_encoder_modules:
        module.train()
    coord_embed = getattr(encoder, "coord_embed", None)
    if coord_embed is not None:
        coord_embed.eval()


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    total_steps = int(total_steps)
    if total_steps <= 0:
        return None

    warmup_steps = int(round(float(warmup_ratio) * float(total_steps)))
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))
    min_lr_ratio = float(min_lr_ratio)

    def _lr_lambda(step_idx: int) -> float:
        step_idx = int(step_idx)
        if warmup_steps > 0 and step_idx < warmup_steps:
            return max(float(step_idx + 1) / float(warmup_steps), 1e-8)
        if total_steps <= warmup_steps + 1:
            return max(min_lr_ratio, 1e-8)
        progress = float(step_idx - warmup_steps) / float(max(total_steps - warmup_steps - 1, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def train_task_finetune(
    encoder: EEGEncoder,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    y_train: torch.Tensor,
    n_classes: int,
    norm_stats: FeatureNormStats,
    lp_head_state: Dict[str, torch.Tensor],
    device: torch.device,
    device_ids: List[int],
    coord_scale: float,
    pool: str,
    amp: bool,
    apply_rescale: bool,
    rescale_kwargs: Dict[str, float],
    backbone_lr: float,
    head_lr_mult: float,
    train_scope: str,
    unfreeze_last_k: int,
    layer_decay: float,
    epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    weight_decay: float,
    class_weight: Optional[str],
    warmup_ratio: float,
    min_lr_ratio: float,
    grad_clip: float,
    seed: int = 42,
) -> FTResult:
    set_seed(seed)

    feature_model = EncoderFeaturizer(
        encoder,
        pool=pool,
        amp=amp,
        apply_rescale=apply_rescale,
        rescale_kwargs=rescale_kwargs,
    )
    model = EEGClassifier(feature_model, n_classes=n_classes, norm_stats=norm_stats)
    model.head.load_state_dict(copy.deepcopy(lp_head_state))
    model.to(device)

    base_model = model
    encoder_param_groups, trainable_encoder_modules, ft_plan = build_ft_encoder_param_groups(
        base_model.feature_model.encoder,
        train_scope=train_scope,
        unfreeze_last_k=unfreeze_last_k,
        backbone_lr=float(backbone_lr),
        layer_decay=float(layer_decay),
        weight_decay=float(weight_decay),
    )
    if not encoder_param_groups:
        raise RuntimeError("No trainable encoder parameter groups were constructed for fine-tuning.")

    head_lr = float(backbone_lr) * float(head_lr_mult)
    head_param_groups = _build_param_groups_for_module(
        base_model.head,
        lr=float(head_lr),
        weight_decay=float(weight_decay),
        group_name="head",
    )
    if not head_param_groups:
        raise RuntimeError("No trainable head parameters found for fine-tuning.")

    optimizer = torch.optim.AdamW(encoder_param_groups + head_param_groups)
    trainable_params_for_clip = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    base_model = unwrap_module(model)

    if class_weight == "balanced":
        with torch.no_grad():
            counts = torch.bincount(y_train.cpu(), minlength=n_classes).float()
            w = (counts.sum() / counts.clamp_min(1.0))
            w = (w / w.mean()).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=w)
    else:
        loss_fn = nn.CrossEntropyLoss()

    train_loader = _make_eeg_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        coord_scale=coord_scale,
    )
    val_loader = _make_eeg_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        coord_scale=coord_scale,
    )
    test_loader = _make_eeg_loader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        coord_scale=coord_scale,
    )

    total_steps = int(max(1, epochs * max(len(train_loader), 1)))
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )

    best_val = -1.0
    best_epoch = -1
    best_test_acc = -1.0
    best_val_f1w = -1.0
    best_test_f1w = -1.0
    best_test_kappa = -1.0
    train_acc_at_best = 0.0
    train_acc_last = 0.0
    last_epoch = 0
    bad = 0

    best_encoder_state = cpu_clone_state_dict(base_model.feature_model.encoder)
    best_head_state = cpu_clone_state_dict(base_model.head)

    autocast = torch.amp.autocast

    for ep in range(1, epochs + 1):
        model.train()
        _set_task_ft_train_mode(base_model, trainable_encoder_modules)
        correct_tr = 0
        total_tr = 0

        for eeg, coord, yb in train_loader:
            eeg = eeg.to(device, non_blocking=True)
            coord = coord.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, dtype=torch.bfloat16, enabled=(amp and device.type == "cuda")):
                logits = model(eeg, coord)
                loss = loss_fn(logits, yb)

            pred = logits.argmax(dim=-1)
            correct_tr += int((pred == yb).sum().item())
            total_tr += int(yb.numel())

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params_for_clip, max_norm=float(grad_clip))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        train_acc = correct_tr / max(1, total_tr)
        train_acc_last = float(train_acc)
        last_epoch = int(ep)

        with torch.no_grad():
            val_metrics = evaluate_classifier(
                model,
                val_loader,
                n_classes=n_classes,
                device=device,
                binary_test_metrics=False,
            )

        if is_better_candidate(val_metrics.acc, val_metrics.f1w, best_val, best_val_f1w):
            with torch.no_grad():
                test_metrics = evaluate_classifier(
                    model,
                    test_loader,
                    n_classes=n_classes,
                    device=device,
                    binary_test_metrics=(n_classes == 2),
                )
            best_val = float(val_metrics.acc)
            best_epoch = int(ep)
            best_test_acc = float(test_metrics.acc)
            best_val_f1w = float(val_metrics.f1w)
            best_test_f1w = float(test_metrics.f1w)
            best_test_kappa = float(test_metrics.kappa)
            train_acc_at_best = float(train_acc_last)
            best_encoder_state = cpu_clone_state_dict(base_model.feature_model.encoder)
            best_head_state = cpu_clone_state_dict(base_model.head)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    base_model.feature_model.encoder.load_state_dict(best_encoder_state)
    base_model.head.load_state_dict(best_head_state)

    return FTResult(
        best_epoch=best_epoch,
        best_val_acc=best_val,
        test_acc_at_best=best_test_acc,
        best_val_f1w=best_val_f1w,
        test_f1w_at_best=best_test_f1w,
        test_kappa_at_best=best_test_kappa,
        backbone_lr=float(backbone_lr),
        head_lr=float(head_lr),
        train_acc_at_best=float(train_acc_at_best),
        train_acc_last=float(train_acc_last),
        last_epoch=int(last_epoch),
        train_scope=str(ft_plan["train_scope"]),
        unfreeze_last_k=int(ft_plan["unfreeze_last_k"]),
        trainable_block_indices=tuple(int(v) for v in ft_plan["trainable_block_indices"]),
        encoder_trainable_params=int(ft_plan["encoder_trainable_params"]),
        encoder_total_params=int(ft_plan["encoder_total_params"]),
        layer_decay=float(ft_plan["layer_decay"]),
        encoder_state_dict=best_encoder_state,
        head_state_dict=best_head_state,
    )


# -------------------------
# checkpoint save helpers
# -------------------------
def _copy_metadata_files(src_dir: str, dst_dir: Path) -> None:
    src = Path(src_dir).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint source directory not found: {src}")

    copied_any = False
    for pattern in ("*.json", "*.txt"):
        for path in src.glob(pattern):
            if path.is_file():
                shutil.copy2(path, dst_dir / path.name)
                copied_any = True

    cfg = src / "config.json"
    if cfg.exists() and not (dst_dir / "config.json").exists():
        shutil.copy2(cfg, dst_dir / "config.json")
        copied_any = True

    if not copied_any:
        raise FileNotFoundError(f"No metadata files (.json/.txt) found to copy from checkpoint dir: {src}")


def save_encoder_checkpoint(encoder: EEGEncoder, source_ckpt_dir: str, out_dir: str) -> str:
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    base_encoder = unwrap_module(encoder)
    save_pretrained = getattr(base_encoder, "save_pretrained", None)
    used_fallback = False

    if callable(save_pretrained):
        try:
            save_pretrained(str(out_path))
        except Exception as exc:
            print(f"[save] save_pretrained failed for {out_path}: {exc}. Falling back to pytorch_model.bin export.")
            used_fallback = True
    else:
        used_fallback = True

    if used_fallback:
        _copy_metadata_files(source_ckpt_dir, out_path)
        torch.save(cpu_clone_state_dict(base_encoder), out_path / "pytorch_model.bin")

    if not (out_path / "config.json").exists():
        _copy_metadata_files(source_ckpt_dir, out_path)

    if not (out_path / "pytorch_model.bin").exists():
        torch.save(cpu_clone_state_dict(base_encoder), out_path / "pytorch_model.bin")

    return str(out_path)


def save_task_bundle(bundle: Dict[str, object], bundle_path: str) -> str:
    out_path = Path(bundle_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    return str(out_path)


# -------------------------
# Parser / entrypoint
# -------------------------
def build_parser(
    add_help: bool = True,
    *,
    include_wandb: bool = True,
    include_seed: bool = True,
) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=add_help)

    ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Base pretrained checkpoint dir containing config.json + pytorch_model.bin. "
             "This checkpoint is used both as the initial cumulative model (model 1) and as the frozen LP source (model 2).",
    )

    ap.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of task names to train. If omitted, train every discovered/explicit task.",
    )
    ap.add_argument(
        "--task_root",
        type=str,
        default="",
        help="Root directory to recursively scan for *_npy task folders containing train/ and test/ splits.",
    )
    ap.add_argument("--tuab_root", type=str, default="", help="Explicit TUAB cache root containing train/[val]/test")
    ap.add_argument("--isruc_root", type=str, default="", help="Explicit ISRUC cache root containing train/[val]/test")
    ap.add_argument("--mi_root", type=str, default="", help="Explicit PhysioNetMI cache root containing train/[val]/test")
    ap.add_argument(
        "--missing_val_ratio",
        type=float,
        default=0.2,
        help="When val/ is missing, split this fraction of train into val using a stratified split.",
    )
    if include_seed:
        ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--save_root", type=str, default="./ft_runs", help="Directory to save per-task checkpoints and bundles.")
    ap.add_argument("--feat_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4, help="Start with 0 (safe). Increase to 2/4 if stable.")
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--num_gpus", type=int, default=1, choices=[1, 2], help="Number of GPUs to use for feature extraction / fine-tuning (1 or 2).")
    add_bool_arg(ap, "amp", default=True, help_text="Enable AMP for feature extraction and fine-tuning.")

    ap.add_argument(
        "--pool",
        type=str,
        default="mean_std",
        choices=["mean", "mean_std", "tc_mean_std", "ct_mean_std", "tc_ct_mean_std"],
        help="Token pooling for classifier features. mean_std is more robust than mean for large token counts.",
    )
    ap.add_argument(
        "--feat_norm",
        type=str,
        default="zscore",
        choices=["none", "zscore", "l2"],
        help="Normalize pooled features using frozen train-set statistics from the LP stage.",
    )
    ap.add_argument(
        "--coord_scale",
        type=float,
        default=10.0,
        help="Multiply coords by this factor before feeding the model. Keep this aligned with how downstream npy coords were exported.",
    )

    add_bool_arg(ap, "apply_rescale", default=True, help_text="Apply the same robust amplitude rescale used in train.py (recommended).")
    ap.add_argument("--rescale_target_amp", type=float, default=1.0)
    ap.add_argument("--rescale_quantile", type=float, default=0.90)
    ap.add_argument("--rescale_amp_floor", type=float, default=1e-4)
    ap.add_argument("--rescale_gain_max", type=float, default=200.0)
    ap.add_argument("--rescale_clip", type=float, default=15.0)

    # LP stage
    ap.add_argument("--epochs", type=int, default=100, help="LP epochs.")
    ap.add_argument("--patience", type=int, default=10, help="LP early-stopping patience.")
    ap.add_argument("--lp_batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3, help="LP LR used if --lrs is not provided.")
    ap.add_argument("--lrs", type=float, nargs="*", default=None, help="LP LR grid (AdamW).")
    ap.add_argument("--class_weight", type=str, default="balanced", choices=["none", "balanced"])

    # FT stage
    ap.add_argument("--ft_epochs", type=int, default=None, help="Fine-tuning epochs. Defaults to --epochs.")
    ap.add_argument("--ft_patience", type=int, default=None, help="Fine-tuning early-stopping patience. Defaults to --patience.")
    ap.add_argument("--ft_batch_size", type=int, default=32)
    ap.add_argument("--ft_lr", type=float, default=3e-5, help="Backbone LR used if --ft_lrs is not provided.")
    ap.add_argument("--ft_lrs", type=float, nargs="*", default=None, help="Backbone LR grid. Typically smaller than LP LR.")
    ap.add_argument("--ft_head_lr_mult", type=float, default=10.0, help="Fine-tuning head LR = ft_lr * ft_head_lr_mult.")
    ap.add_argument(
        "--ft_train_scope",
        type=str,
        default="full",
        choices=["last_k", "full"],
        help="Fine-tuning trainable encoder scope.",
    )
    ap.add_argument(
        "--ft_unfreeze_last_k",
        type=int,
        default=0,
        help="When --ft_train_scope=last_k, unfreeze this many final encoder blocks. <=0 uses an automatic heuristic.",
    )
    ap.add_argument(
        "--ft_layer_decay",
        type=float,
        default=0.75,
        help="Layer-wise LR decay across trainable encoder modules. 1.0 disables layer-wise decay.",
    )
    ap.add_argument("--ft_weight_decay", type=float, default=0.05)
    ap.add_argument("--ft_warmup_ratio", type=float, default=0.1)
    ap.add_argument("--ft_min_lr_ratio", type=float, default=0.0)
    ap.add_argument("--ft_grad_clip", type=float, default=1.0, help="Set <=0 to disable gradient clipping during fine-tuning.")

    if include_wandb:
        ap.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging.")
        ap.add_argument("--wandb_project", type=str, default="EEG_FM")
        ap.add_argument("--wandb_name", type=str, default="")

    ap.add_argument("--out_csv", type=str, default=None, help="Optional path to output CSV summary. Defaults to <save_root>/summary.csv.")
    return ap


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def select_device(num_gpus: int) -> Tuple[torch.device, List[int]]:
    if not torch.cuda.is_available():
        return torch.device("cpu"), []

    available = torch.cuda.device_count()
    requested = int(num_gpus)
    if requested > available:
        raise ValueError(f"Requested num_gpus={requested}, but only {available} CUDA device(s) are visible.")
    device_ids = list(range(requested))
    return torch.device(f"cuda:{device_ids[0]}"), device_ids


def save_run_config(args: argparse.Namespace, task_specs: List[Tuple[str, str]], save_root: Path) -> str:
    payload = dict(vars(args))
    payload["task_specs"] = [{"task": task, "root": root} for task, root in task_specs]
    path = save_root / "run_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(path)


def run_sequential_training(args: argparse.Namespace) -> List[Dict[str, object]]:
    seed = int(getattr(args, "seed", 42))
    set_seed(seed)

    if float(args.ft_head_lr_mult) <= 0.0:
        raise ValueError("--ft_head_lr_mult must be > 0.")
    if not (0.0 < float(args.ft_layer_decay) <= 1.0):
        raise ValueError("--ft_layer_decay must satisfy 0 < value <= 1.")
    if not (0.0 <= float(args.ft_warmup_ratio) < 1.0):
        raise ValueError("--ft_warmup_ratio must satisfy 0 <= value < 1.")
    if not (0.0 <= float(args.ft_min_lr_ratio) <= 1.0):
        raise ValueError("--ft_min_lr_ratio must satisfy 0 <= value <= 1.")

    ft_epochs = int(args.ft_epochs) if args.ft_epochs is not None else int(args.epochs)
    ft_patience = int(args.ft_patience) if args.ft_patience is not None else int(args.patience)

    device, device_ids = select_device(args.num_gpus)
    task_specs = resolve_task_specs(args)
    ckpt_dir = str(Path(args.ckpt).expanduser())
    save_root = Path(args.save_root).expanduser()
    save_root.mkdir(parents=True, exist_ok=True)
    run_config_path = save_run_config(args, task_specs, save_root)

    print(
        f"[train] device={device} visible_cuda={torch.cuda.device_count()} "
        f"num_gpus_used={len(device_ids) if device.type == 'cuda' else 0} "
        f"amp={args.amp} num_workers={args.num_workers} pin_memory={args.pin_memory}"
    )
    print(f"[train] ckpt={ckpt_dir}")
    print(f"[train] save_root={save_root}")
    print(f"[train] run_config={run_config_path}")
    print(f"[train] tasks={', '.join([f'{task}:{root}' for task, root in task_specs])}")
    print(
        f"[train] fine-tune config: ft_epochs={ft_epochs} ft_patience={ft_patience} "
        f"ft_batch_size={args.ft_batch_size} ft_train_scope={args.ft_train_scope} "
        f"ft_unfreeze_last_k={args.ft_unfreeze_last_k} ft_layer_decay={args.ft_layer_decay} "
        f"ft_head_lr_mult={args.ft_head_lr_mult} ft_weight_decay={args.ft_weight_decay} "
        f"ft_warmup_ratio={args.ft_warmup_ratio} ft_min_lr_ratio={args.ft_min_lr_ratio} "
        f"ft_grad_clip={args.ft_grad_clip}"
    )

    use_wandb = (wandb is not None) and (not bool(getattr(args, "no_wandb", False)))
    if use_wandb:
        run_name = getattr(args, "wandb_name", "") or None
        wandb.init(
            project=getattr(args, "wandb_project", "EEG_FM"),
            name=run_name,
            config={
                "mode": "sequential_task_ft",
                "ckpt": ckpt_dir,
                "tasks": [task for task, _ in task_specs],
                "seed": seed,
                "save_root": str(save_root),
                "feat_batch_size": args.feat_batch_size,
                "lp_batch_size": args.lp_batch_size,
                "lp_epochs": args.epochs,
                "lp_patience": args.patience,
                "lp_lr_grid": (args.lrs if args.lrs else [args.lr]),
                "ft_epochs": ft_epochs,
                "ft_patience": ft_patience,
                "ft_batch_size": args.ft_batch_size,
                "ft_lr_grid": (args.ft_lrs if args.ft_lrs else [args.ft_lr]),
                "ft_train_scope": args.ft_train_scope,
                "ft_unfreeze_last_k": args.ft_unfreeze_last_k,
                "ft_layer_decay": args.ft_layer_decay,
                "ft_head_lr_mult": args.ft_head_lr_mult,
                "ft_weight_decay": args.ft_weight_decay,
                "ft_warmup_ratio": args.ft_warmup_ratio,
                "ft_min_lr_ratio": args.ft_min_lr_ratio,
                "ft_grad_clip": args.ft_grad_clip,
                "class_weight": args.class_weight,
                "num_gpus": args.num_gpus,
                "missing_val_ratio": args.missing_val_ratio,
                "apply_rescale": args.apply_rescale,
                "rescale_target_amp": args.rescale_target_amp,
                "rescale_quantile": args.rescale_quantile,
                "rescale_amp_floor": args.rescale_amp_floor,
                "rescale_gain_max": args.rescale_gain_max,
                "rescale_clip": args.rescale_clip,
            },
        )

    lp_lr_grid = args.lrs if args.lrs and len(args.lrs) > 0 else [args.lr]
    ft_lr_grid = args.ft_lrs if args.ft_lrs and len(args.ft_lrs) > 0 else [args.ft_lr]

    rescale_kwargs = dict(
        target_amp=float(args.rescale_target_amp),
        quantile=float(args.rescale_quantile),
        amp_floor=float(args.rescale_amp_floor),
        gain_max=float(args.rescale_gain_max),
        clip=float(args.rescale_clip),
    )

    cumulative_encoder = EEGEncoder.from_pretrained(ckpt_dir, map_location="cpu")
    cumulative_encoder.eval()

    csv_rows: List[Dict[str, object]] = []

    for task_idx, (task, root) in enumerate(task_specs, start=1):
        print()
        print(f"[task={task}] ({task_idx}/{len(task_specs)}) loading splits from {root}")
        train_ds, val_ds, test_ds, n_classes, split_info = load_task_splits(
            task=task,
            root=root,
            seed=seed,
            missing_val_ratio=args.missing_val_ratio,
        )
        print(
            f"[task={task}] val_source={split_info.val_source} labels={list(split_info.label_values)} "
            f"n_classes={n_classes}"
        )

        # ---------------------------------
        # Stage A: LP search on frozen model 2
        # ---------------------------------
        print(f"[task={task}] [LP] loading frozen source encoder from {ckpt_dir}")
        cumulative_encoder.to("cpu")
        maybe_empty_cache()

        frozen_encoder = EEGEncoder.from_pretrained(ckpt_dir, map_location="cpu")
        frozen_encoder.to(device)
        frozen_encoder.eval()

        frozen_feature_model: nn.Module = EncoderFeaturizer(
            frozen_encoder,
            pool=args.pool,
            amp=args.amp,
            apply_rescale=args.apply_rescale,
            rescale_kwargs=rescale_kwargs,
        )
        frozen_feature_model.to(device)
        frozen_feature_model.eval()

        if len(device_ids) > 1:
            frozen_feature_model = nn.DataParallel(frozen_feature_model, device_ids=device_ids, output_device=device_ids[0])
            print(f"[task={task}] [LP] feature extraction will use DataParallel on GPUs={device_ids}")

        print(f"[task={task}] [LP] extracting features: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
        Xtr_raw, ytr = extract_features(
            feature_model=frozen_feature_model,
            ds=train_ds,
            device=device,
            feat_batch_size=args.feat_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            coord_scale=args.coord_scale,
        )
        Xva_raw, yva = extract_features(
            feature_model=frozen_feature_model,
            ds=val_ds,
            device=device,
            feat_batch_size=args.feat_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            coord_scale=args.coord_scale,
        )
        Xte_raw, yte = extract_features(
            feature_model=frozen_feature_model,
            ds=test_ds,
            device=device,
            feat_batch_size=args.feat_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            coord_scale=args.coord_scale,
        )

        del frozen_feature_model
        del frozen_encoder
        gc.collect()
        maybe_empty_cache()

        norm_stats = compute_feature_norm_stats(Xtr_raw, mode=args.feat_norm)
        Xtr, Xva, Xte = normalize_features(Xtr_raw, Xva_raw, Xte_raw, stats=norm_stats)

        with torch.no_grad():
            counts = torch.bincount(ytr.cpu(), minlength=n_classes).float()
            maj = float((counts.max() / counts.sum().clamp_min(1.0)).item())
            feat_std_mean = float(Xtr.std(dim=0).mean().item())
            feat_abs_mean = float(Xtr.abs().mean().item())
        print(
            f"[task={task}] [LP] pool={args.pool} feat_norm={args.feat_norm} |X|mean={feat_abs_mean:.4f} "
            f"std_dim_mean={feat_std_mean:.4f} maj={maj:.3f} counts={counts.tolist()}"
        )

        best_lp: Optional[LPResult] = None
        for lp_lr in lp_lr_grid:
            lp_result = train_linear_probe(
                X_train=Xtr,
                y_train=ytr,
                X_val=Xva,
                y_val=yva,
                X_test=Xte,
                y_test=yte,
                n_classes=n_classes,
                device=device,
                lr=float(lp_lr),
                epochs=int(args.epochs),
                patience=int(args.patience),
                batch_size=int(args.lp_batch_size),
                weight_decay=0.0,
                class_weight=None if args.class_weight == "none" else "balanced",
                seed=seed,
            )
            print(
                f"[task={task}] [LP] lr={lp_lr:.2e} best_val={lp_result.best_val_acc:.4f} "
                f"test_acc={lp_result.test_acc_at_best:.4f} test_f1w={lp_result.test_f1w_at_best:.4f} "
                f"test_kappa={lp_result.test_kappa_at_best:.4f} best_ep={lp_result.best_epoch} "
                f"train_last={lp_result.train_acc_last:.4f} train@best={lp_result.train_acc_at_best:.4f} "
                f"stop_ep={lp_result.last_epoch}"
            )
            if (best_lp is None) or is_better_candidate(
                lp_result.best_val_acc,
                lp_result.best_val_f1w,
                best_lp.best_val_acc,
                best_lp.best_val_f1w,
            ):
                best_lp = lp_result

        assert best_lp is not None and best_lp.state_dict is not None
        print(
            f"[task={task}] BEST[LP]: val_acc={best_lp.best_val_acc:.4f} val_f1w={best_lp.best_val_f1w:.4f} "
            f"test_acc={best_lp.test_acc_at_best:.4f} test_f1w={best_lp.test_f1w_at_best:.4f} "
            f"test_kappa={best_lp.test_kappa_at_best:.4f} lr={best_lp.lr:.2e}"
        )

        # ---------------------------------
        # Stage B: cumulative FT on model 1
        # ---------------------------------
        pre_task_encoder_state = cpu_clone_state_dict(cumulative_encoder)
        best_ft: Optional[FTResult] = None

        for ft_lr in ft_lr_grid:
            cumulative_encoder.load_state_dict(pre_task_encoder_state)
            cumulative_encoder.to("cpu")
            maybe_empty_cache()

            ft_result = train_task_finetune(
                encoder=cumulative_encoder,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                y_train=ytr,
                n_classes=n_classes,
                norm_stats=norm_stats,
                lp_head_state=best_lp.state_dict,
                device=device,
                device_ids=device_ids,
                coord_scale=args.coord_scale,
                pool=args.pool,
                amp=args.amp,
                apply_rescale=args.apply_rescale,
                rescale_kwargs=rescale_kwargs,
                backbone_lr=float(ft_lr),
                head_lr_mult=float(args.ft_head_lr_mult),
                train_scope=str(args.ft_train_scope),
                unfreeze_last_k=int(args.ft_unfreeze_last_k),
                layer_decay=float(args.ft_layer_decay),
                epochs=ft_epochs,
                patience=ft_patience,
                batch_size=int(args.ft_batch_size),
                num_workers=int(args.num_workers),
                pin_memory=bool(args.pin_memory),
                weight_decay=float(args.ft_weight_decay),
                class_weight=None if args.class_weight == "none" else "balanced",
                warmup_ratio=float(args.ft_warmup_ratio),
                min_lr_ratio=float(args.ft_min_lr_ratio),
                grad_clip=float(args.ft_grad_clip),
                seed=seed,
            )
            print(
                f"[task={task}] [FT] scope={ft_result.train_scope} unfreeze_last_k={ft_result.unfreeze_last_k} "
                f"blocks={list(ft_result.trainable_block_indices)} "
                f"trainable={ft_result.encoder_trainable_params/1e6:.2f}M/"
                f"{ft_result.encoder_total_params/1e6:.2f}M backbone_lr={ft_lr:.2e} "
                f"head_lr={ft_result.head_lr:.2e} layer_decay={ft_result.layer_decay:.3f} "
                f"best_val={ft_result.best_val_acc:.4f} test_acc={ft_result.test_acc_at_best:.4f} "
                f"test_f1w={ft_result.test_f1w_at_best:.4f} test_kappa={ft_result.test_kappa_at_best:.4f} "
                f"best_ep={ft_result.best_epoch} train_last={ft_result.train_acc_last:.4f} "
                f"train@best={ft_result.train_acc_at_best:.4f} stop_ep={ft_result.last_epoch}"
            )
            if (best_ft is None) or is_better_candidate(
                ft_result.best_val_acc,
                ft_result.best_val_f1w,
                best_ft.best_val_acc,
                best_ft.best_val_f1w,
            ):
                best_ft = ft_result

            cumulative_encoder.to("cpu")
            gc.collect()
            maybe_empty_cache()

        assert best_ft is not None
        cumulative_encoder.load_state_dict(best_ft.encoder_state_dict)
        cumulative_encoder.to("cpu")

        task_tag = f"{task_idx:02d}_{normalize_task_name(task)}"
        task_dir = save_root / task_tag
        encoder_ckpt_dir = task_dir / "encoder"
        bundle_path = task_dir / "task_bundle.pt"

        saved_encoder_ckpt = save_encoder_checkpoint(cumulative_encoder, ckpt_dir, str(encoder_ckpt_dir))
        saved_bundle = save_task_bundle(
            {
                "task": task,
                "task_index": int(task_idx),
                "task_root": root,
                "source_pretrained_ckpt": ckpt_dir,
                "label_values": tuple(int(v) for v in split_info.label_values),
                "n_classes": int(n_classes),
                "val_source": split_info.val_source,
                "pool": str(args.pool),
                "feat_norm": str(args.feat_norm),
                "norm_stats": serialize_feature_norm_stats(norm_stats),
                "coord_scale": float(args.coord_scale),
                "apply_rescale": bool(args.apply_rescale),
                "rescale_kwargs": dict(rescale_kwargs),
                "lp": {
                    "lr": float(best_lp.lr),
                    "best_epoch": int(best_lp.best_epoch),
                    "stop_epoch": int(best_lp.last_epoch),
                    "train_acc_last": float(best_lp.train_acc_last),
                    "train_acc_at_best": float(best_lp.train_acc_at_best),
                    "val_acc": float(best_lp.best_val_acc),
                    "val_f1w": float(best_lp.best_val_f1w),
                    "test_acc": float(best_lp.test_acc_at_best),
                    "test_f1w": float(best_lp.test_f1w_at_best),
                    "test_kappa": float(best_lp.test_kappa_at_best),
                    "head_state_dict": best_lp.state_dict,
                },
                "ft": {
                    "backbone_lr": float(best_ft.backbone_lr),
                    "head_lr": float(best_ft.head_lr),
                    "best_epoch": int(best_ft.best_epoch),
                    "stop_epoch": int(best_ft.last_epoch),
                    "train_scope": str(best_ft.train_scope),
                    "unfreeze_last_k": int(best_ft.unfreeze_last_k),
                    "trainable_block_indices": tuple(int(v) for v in best_ft.trainable_block_indices),
                    "encoder_trainable_params": int(best_ft.encoder_trainable_params),
                    "encoder_total_params": int(best_ft.encoder_total_params),
                    "layer_decay": float(best_ft.layer_decay),
                    "train_acc_last": float(best_ft.train_acc_last),
                    "train_acc_at_best": float(best_ft.train_acc_at_best),
                    "val_acc": float(best_ft.best_val_acc),
                    "val_f1w": float(best_ft.best_val_f1w),
                    "test_acc": float(best_ft.test_acc_at_best),
                    "test_f1w": float(best_ft.test_f1w_at_best),
                    "test_kappa": float(best_ft.test_kappa_at_best),
                    "head_state_dict": best_ft.head_state_dict,
                },
            },
            str(bundle_path),
        )

        row = {
            "task_index": int(task_idx),
            "task": task,
            "task_root": root,
            "val_source": split_info.val_source,
            "label_values": ";".join(str(v) for v in split_info.label_values),
            "n_classes": int(n_classes),
            "lp_lr": float(best_lp.lr),
            "lp_best_epoch": int(best_lp.best_epoch),
            "lp_stop_epoch": int(best_lp.last_epoch),
            "lp_train_acc_last": float(best_lp.train_acc_last),
            "lp_train_acc_at_best": float(best_lp.train_acc_at_best),
            "lp_val_acc": float(best_lp.best_val_acc),
            "lp_val_f1w": float(best_lp.best_val_f1w),
            "lp_test_acc": float(best_lp.test_acc_at_best),
            "lp_test_f1w": float(best_lp.test_f1w_at_best),
            "lp_test_kappa": float(best_lp.test_kappa_at_best),
            "ft_lr": float(best_ft.backbone_lr),
            "ft_head_lr": float(best_ft.head_lr),
            "ft_best_epoch": int(best_ft.best_epoch),
            "ft_stop_epoch": int(best_ft.last_epoch),
            "ft_train_scope": str(best_ft.train_scope),
            "ft_unfreeze_last_k": int(best_ft.unfreeze_last_k),
            "ft_trainable_blocks": ";".join(str(v) for v in best_ft.trainable_block_indices),
            "ft_encoder_trainable_params": int(best_ft.encoder_trainable_params),
            "ft_encoder_total_params": int(best_ft.encoder_total_params),
            "ft_layer_decay": float(best_ft.layer_decay),
            "ft_train_acc_last": float(best_ft.train_acc_last),
            "ft_train_acc_at_best": float(best_ft.train_acc_at_best),
            "ft_val_acc": float(best_ft.best_val_acc),
            "ft_val_f1w": float(best_ft.best_val_f1w),
            "ft_test_acc": float(best_ft.test_acc_at_best),
            "ft_test_f1w": float(best_ft.test_f1w_at_best),
            "ft_test_kappa": float(best_ft.test_kappa_at_best),
            "saved_encoder_ckpt": saved_encoder_ckpt,
            "saved_task_bundle": saved_bundle,
        }
        csv_rows.append(row)

        print(
            f"[task={task}] BEST[FT]: val_acc={best_ft.best_val_acc:.4f} val_f1w={best_ft.best_val_f1w:.4f} "
            f"test_acc={best_ft.test_acc_at_best:.4f} test_f1w={best_ft.test_f1w_at_best:.4f} "
            f"test_kappa={best_ft.test_kappa_at_best:.4f} scope={best_ft.train_scope} "
            f"unfreeze_last_k={best_ft.unfreeze_last_k} blocks={list(best_ft.trainable_block_indices)} "
            f"trainable={best_ft.encoder_trainable_params/1e6:.2f}M/{best_ft.encoder_total_params/1e6:.2f}M "
            f"backbone_lr={best_ft.backbone_lr:.2e} head_lr={best_ft.head_lr:.2e} "
            f"layer_decay={best_ft.layer_decay:.3f}"
        )
        print(f"[task={task}] saved encoder checkpoint: {saved_encoder_ckpt}")
        print(f"[task={task}] saved task bundle: {saved_bundle}")

        if use_wandb:
            wandb.log({
                "task_index": int(task_idx),
                "task_name": task,
                "lp/lr": float(best_lp.lr),
                "lp/val_acc": float(best_lp.best_val_acc),
                "lp/test_acc": float(best_lp.test_acc_at_best),
                "ft/backbone_lr": float(best_ft.backbone_lr),
                "ft/head_lr": float(best_ft.head_lr),
                "ft/val_acc": float(best_ft.best_val_acc),
                "ft/test_acc": float(best_ft.test_acc_at_best),
            })

        gc.collect()
        maybe_empty_cache()

    final_encoder_dir = save_root / "final_encoder"
    cumulative_encoder.to("cpu")
    final_encoder_ckpt = save_encoder_checkpoint(cumulative_encoder, ckpt_dir, str(final_encoder_dir))
    print()
    print(f"[train] final cumulative encoder checkpoint: {final_encoder_ckpt}")

    out_csv = Path(args.out_csv).expanduser() if args.out_csv else (save_root / "summary.csv")
    if csv_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        print(f"[train] wrote csv summary: {out_csv}")

    if use_wandb and csv_rows:
        cols = list(csv_rows[0].keys())
        table = wandb.Table(columns=cols)
        for row in csv_rows:
            table.add_data(*[row.get(c, None) for c in cols])
        wandb.log({"sequential_train_results": table})
        wandb.finish()

    print()
    print("[train] done.")
    return csv_rows


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    return run_sequential_training(parse_args(argv))


if __name__ == "__main__":
    main()
