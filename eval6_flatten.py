"""
eval_linear_probe_npy.py

Linear-probe / LPFT evaluation for EEG foundation model checkpoints using cached .npy arrays (memmap).
This script assumes each dataset split was extracted into:
  <DATASET_CACHE_ROOT>/<split>/eeg.npy
  <DATASET_CACHE_ROOT>/<split>/coords.npy
  <DATASET_CACHE_ROOT>/<split>/label.npy

Key features:
  - Recursively scan --task_root for *_npy task folders.
  - Evaluate every discovered task (or a subset via --tasks).
  - Use provided train/val/test when available; if val/ is missing, create it from train by stratified split.
  - Test metrics always include accuracy.
    * Multi-class: test_f1w=weighted F1, test_kappa=Cohen's kappa.
    * Binary: test_f1w=AUC-PR, test_kappa=AUROC.
  - Feature extraction and fine-tuning can run on 1 or 2 GPUs via DataParallel (--num_gpus).
  - Default mode is LP only. Add --lpft to run LP first, then fine-tune the classifier with a
    recommended partial-FT recipe: unfreeze only the final encoder blocks + final norm by default.
  - Optional LoRA adapters can be enabled for encoder attention QKV/O projections via --ft_lora.
    LoRA respects the same last_k/full scope selection and freezes the base encoder weights.
  - LP and FT learning rates are tuned separately per task. FT uses discriminative learning rates
    (smaller encoder LR, larger head LR), layer-wise LR decay, cosine decay with linear warmup,
    and gradient clipping. Use --ft_train_scope full to recover full-encoder FT.

Examples:
  CUDA_VISIBLE_DEVICES=0 python -m eeg_fm.eval \
    --ckpt /data/ssd/checkpoints/A2_1_B/final/teacher \
    --task_root /mnt/e/downstream_tasks \
    --num_gpus 1 \
    --out_csv /data/ssd/eval_results/eval.csv

  CUDA_VISIBLE_DEVICES=0,1 python -m eeg_fm.eval \
    --ckpt /data/ssd/checkpoints/A2_1_B/final/teacher \
    --task_root /mnt/e/downstream_tasks \
    --num_gpus 2 \
    --lpft \
    --lrs 3e-4 1e-3 3e-3 1e-2 \
    --ft_lrs 5e-6 1e-5 3e-5 1e-4 \
    --ft_head_lr_mult 10.0 \
    --out_csv /data/ssd/eval_results/eval_lpft.csv

  CUDA_VISIBLE_DEVICES=0 python -m eeg_fm.eval \
    --ckpt /data/ssd/checkpoints/A2_1_B/final/teacher \
    --task_root /mnt/e/downstream_tasks \
    --lpft --ft_lora --ft_lora_rank 8 --ft_lora_alpha 16
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

# Import your encoder
# Expected: EEGEncoder.from_pretrained(<dir with config.json + pytorch_model.bin>)
from .model import EEGEncoder

try:
    import wandb
except Exception:
    wandb = None


# -------------------------
# utils
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
    l2sp_weight: float
    use_lora: bool
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    lora_include_spatial_qk: bool
    lora_wrapped_linears: int
    head_state_dict: Optional[Dict[str, torch.Tensor]] = None
    encoder_state_dict: Optional[Dict[str, torch.Tensor]] = None


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

        # mmap: header + lazy pages
        self.eeg = np.load(eeg_path, mmap_mode="r")
        self.coords = np.load(coords_path, mmap_mode="r")
        self.y = np.load(label_path, mmap_mode="r")

        if self.eeg.shape[0] != self.coords.shape[0] or self.eeg.shape[0] != self.y.shape[0]:
            raise ValueError(f"N mismatch: eeg={self.eeg.shape} coords={self.coords.shape} y={self.y.shape}")

        # Expect (N,C,T) and (N,C,3)
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


class EEGClassifier(nn.Module):
    def __init__(self, feature_model: EncoderFeaturizer, n_classes: int, norm_stats: FeatureNormStatsm, feat_dim2: Optional[int] = None):
        super().__init__()
        self.feature_model = feature_model
        self.feature_norm = FeatureNormalizer(norm_stats)
        # self.head = nn.Linear(int(feature_model.feat_dim), int(n_classes))
        self.head = EEGHead(512*feat_dim2, int(n_classes))

    def forward(self, eeg: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        feat = self.feature_model(eeg, coord)
        feat = self.feature_norm(feat)
        return self.head(feat)

class EEGHead(nn.Module):
    def __init__(self, feat_dim:int, n_classes: int, layers=2):
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
                nn.LayerNorm(int(feat_dim // 4)),
                nn.ELU(),
                nn.Linear(int(feat_dim // 4), int(n_classes)),
            )
        else:
            self.head = nn.Linear(int(feat_dim), int(n_classes))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)

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
    return path.is_dir() and path.name.lower().endswith("_npy") and split_dir_has_required_files(path / "train") and split_dir_has_required_files(path / "test")


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


def resolve_ckpt_dirs(args: argparse.Namespace) -> List[str]:
    ckpts = getattr(args, "ckpts", None)
    if ckpts:
        return [str(Path(p).expanduser()) for p in ckpts]
    ckpt = getattr(args, "ckpt", None)
    if ckpt:
        return [str(Path(ckpt).expanduser())]
    raise ValueError("Missing checkpoint path. Provide --ckpt or --ckpts.")


def compute_num_patches(T: int, patch_samples: int, hop_samples: int) -> int:
    if T < patch_samples:
        return 0
    return int((T - patch_samples) // hop_samples + 1)


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


@torch.no_grad()
def _confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int, device: torch.device) -> torch.Tensor:
    """Confusion matrix (K,K). Rows=true, Cols=pred."""
    y_true = y_true.to(device=device, dtype=torch.long)
    y_pred = y_pred.to(device=device, dtype=torch.long)
    idx = y_true * n_classes + y_pred
    cm = torch.bincount(idx, minlength=n_classes * n_classes)
    return cm.view(n_classes, n_classes)


@torch.no_grad()
def weighted_f1_from_cm(cm: torch.Tensor) -> float:
    """Weighted F1 from confusion matrix."""
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
    # total = float(cmf.sum().item())
    # acc = 0.0 if total <= 0.0 else float(torch.diagonal(cmf).sum().item() / total)
    support = cmf.sum(dim=1)
    per_class_acc = torch.nan_to_num(torch.diagonal(cmf) / support, nan=0.0)
    acc = float(per_class_acc.mean().item())
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
    if binary_test_metrics and n_classes == 2:
        y_true_parts: List[np.ndarray] = []
        y_score_parts: List[np.ndarray] = []
        y_pred_parts: List[np.ndarray] = []
        # correct = 0
        total = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = head(xb)
            pred = logits.argmax(dim=-1)
            scores = (logits[:, 1] - logits[:, 0]).detach().float().cpu().numpy()

            # correct += int((pred == yb).sum().item())
            total += int(yb.numel())
            y_true_parts.append(yb.detach().cpu().numpy())
            y_score_parts.append(scores)
            y_pred_parts.append(pred.detach().cpu().numpy()) # 추가

        if total == 0:
            return Metrics(acc=0.0, f1w=float("nan"), kappa=float("nan"))

        y_true = np.concatenate(y_true_parts, axis=0)
        y_score = np.concatenate(y_score_parts, axis=0)
        y_pred = np.concatenate(y_pred_parts, axis=0)  # 배열로 병합
        # acc = float(correct / total)
        # --- Balanced Accuracy 계산 ---
        # 클래스별 인덱스 추출
        idx_0 = (y_true == 0)
        idx_1 = (y_true == 1)

        # 클래스 0 (Negative)의 정확도 (Specificity)
        acc_0 = (y_pred[idx_0] == 0).sum() / idx_0.sum() if idx_0.sum() > 0 else 0.0
        
        # 클래스 1 (Positive)의 정확도 (Recall/Sensitivity)
        acc_1 = (y_pred[idx_1] == 1).sum() / idx_1.sum() if idx_1.sum() > 0 else 0.0

        # 두 클래스 정확도의 단순 평균 (Balanced Acc)
        acc = float((acc_0 + acc_1) / 2.0)
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


# -------------------------
# Dataset helpers
# -------------------------
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
    """
    Collate factory so coord scaling can be configured from CLI.
    coord_scale should match your pretraining pipeline.
    """
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
    """
    Backward/forward compatible wrapper:
      - Filters kwargs against the current model signature.
      - Accepts both legacy and new model return formats.
    """
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
    """
    Call encoder.forward with only the kwargs it supports.
    This avoids signature mismatch across model versions.
    """
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
    """
    Pool token embeddings into a single feature vector per sample.

    pool:
      - mean: global mean (legacy; can collapse for large L)
      - mean_std: concat(mean, std) over tokens (robust default)
      - tc_mean_std: reshape to (B,P,C,D), pool over C first -> (B,P,D), then concat(time-mean, time-std)
      - ct_mean_std: reshape to (B,P,C,D), pool over P first -> (B,C,D), then concat(chan-mean, chan-std)
      - tc_ct_mean_std: concat(tc_mean_std, ct_mean_std)
    """
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
    """Feature normalization applied after extraction (on CPU tensors)."""
    return (
        apply_feature_norm(X_train, stats),
        apply_feature_norm(X_val, stats),
        apply_feature_norm(X_test, stats),
    )


def clone_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cloned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def module_state_dict_to_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return clone_state_dict_to_cpu(module.state_dict())


def feature_norm_stats_to_payload(stats: FeatureNormStats) -> Dict[str, object]:
    payload: Dict[str, object] = {"mode": str(stats.mode)}
    if stats.mean is not None:
        payload["mean"] = stats.mean.detach().cpu().clone()
    if stats.std is not None:
        payload["std"] = stats.std.detach().cpu().clone()
    return payload


def sanitize_path_component(text: str) -> str:
    text = str(text).strip()
    if not text:
        return "unnamed"
    chars: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            chars.append(ch)
        else:
            chars.append("_")
    safe = "".join(chars).strip("._")
    return safe or "unnamed"


def resolve_lpft_checkpoint_dir(lpft_ckpt_dir: Optional[str], ckpt_dir: str, task: str) -> Path:
    ckpt_name = sanitize_path_component(Path(ckpt_dir).name)
    task_name = sanitize_path_component(task)
    if lpft_ckpt_dir:
        return Path(lpft_ckpt_dir).expanduser() / ckpt_name / task_name
    return Path(ckpt_dir).expanduser() / "eval_lpft_checkpoints" / task_name


def save_lpft_stage_checkpoints(
    output_dir: Path,
    *,
    stage: str,
    ckpt_name: str,
    ckpt_path: str,
    task: str,
    task_root: str,
    label_values: Tuple[int, ...],
    n_classes: int,
    head_state_dict: Dict[str, torch.Tensor],
    encoder_state_dict: Dict[str, torch.Tensor],
    norm_stats: FeatureNormStats,
    metadata: Dict[str, object],
) -> Dict[str, str]:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_payload = {
        "stage": str(stage),
        "ckpt_name": str(ckpt_name),
        "ckpt_path": str(ckpt_path),
        "task": str(task),
        "task_root": str(task_root),
        "label_values": tuple(int(v) for v in label_values),
        "n_classes": int(n_classes),
        "feat_norm": feature_norm_stats_to_payload(norm_stats),
        "metadata": copy.deepcopy(metadata),
    }

    head_path = output_dir / f"best_{stage}_head.pt"
    encoder_path = output_dir / f"best_{stage}_encoder.pt"

    torch.save(
        {
            **base_payload,
            "component": "head",
            "state_dict": clone_state_dict_to_cpu(head_state_dict),
        },
        head_path,
    )
    torch.save(
        {
            **base_payload,
            "component": "encoder",
            "state_dict": clone_state_dict_to_cpu(encoder_state_dict),
        },
        encoder_path,
    )

    return {
        "head": str(head_path),
        "encoder": str(encoder_path),
    }


# -------------------------
# Feature extraction
# -------------------------
def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


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
    """
    Returns:
      X: (N,D_out) float32 CPU
      y: (N,) long CPU
    """
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
    feat_dim = int(getattr(base_model, "embed_dim"))
    for eeg, coord, y in loader:
        _, C, S = eeg.shape
        break
    N = len(ds)
    X = torch.empty((N, feat_dim*C*S), dtype=torch.float32)
    y_all = torch.empty((N,), dtype=torch.long)

    offset = 0
    for eeg, coord, y in loader:
        B = eeg.shape[0]
        eeg = eeg.to(device, non_blocking=True)
        coord = coord.to(device, non_blocking=True)
        feat = feature_model(eeg, coord)
        feat_cpu = feat.detach().float().cpu()
        X[offset : offset + B] = torch.flatten(feat_cpu, start_dim=1)
        y_all[offset : offset + B] = y.cpu()
        offset += B

    assert offset == N
    return X, y_all


# -------------------------
# Linear probe training
# -------------------------
def train_linear_probe(
    feature_model: nn.Module,   # 추가
    train_ds: Dataset,          # 변경
    val_ds: Dataset,            # 변경
    test_ds: Dataset,           # 변경
    n_classes: int,
    device: torch.device,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,           # 추가
    pin_memory: bool,           # 추가
    coord_scale: float,         # 추가
    amp: bool,                  # 추가
    weight_decay: float = 0.0,
    class_weight: Optional[str] = None,
    seed: int = 42,
) -> LPResult:
    set_seed(seed)
# 1. Dummy norm_stats (On-the-fly에서는 전체 데이터 통계(zscore)를 미리 내기 어려우므로 none으로 처리)
    norm_stats = FeatureNormStats(mode="none")

    # 3. 데이터 로더 생성
    train_loader = _make_eeg_loader(train_ds, batch_size=batch_size, shuffle=True, seed=seed, num_workers=num_workers, pin_memory=pin_memory, coord_scale=coord_scale)
    val_loader = _make_eeg_loader(val_ds, batch_size=batch_size, shuffle=False, seed=seed, num_workers=num_workers, pin_memory=pin_memory, coord_scale=coord_scale)
    test_loader = _make_eeg_loader(test_ds, batch_size=batch_size, shuffle=False, seed=seed, num_workers=num_workers, pin_memory=pin_memory, coord_scale=coord_scale)

    for eeg, coord, y in train_loader:
        _, C, S = eeg.shape
        break
        
    loss_fn = nn.CrossEntropyLoss() # (Class weight 적용 시 train_ds를 순회하여 bincount 계산 필요)
    
    # 2. Classifier 구성 및 Feature Model 프리징 (LP이므로 Head만 학습)
    model = EEGClassifier(feature_model, n_classes=n_classes, norm_stats=norm_stats, feat_dim2=C*S).to(device)
    for param in model.feature_model.parameters():
        param.requires_grad = False
        
    opt = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)

    # if class_weight == "balanced":
    #     with torch.no_grad():
    #         counts = torch.bincount(y_train.cpu(), minlength=n_classes).float()
    #         w = (counts.sum() / counts.clamp_min(1.0))
    #         w = (w / w.mean()).to(device)
    #     loss_fn = nn.CrossEntropyLoss(weight=w)
    # else:
    # loss_fn = nn.CrossEntropyLoss()

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
        model.train()
        model.feature_model.eval() # Encoder는 항상 eval 모드
        
        correct_tr = 0
        total_tr = 0
        
        # 4. Train Loop (On-the-fly Feature Extraction)
        for eeg, coord, yb in train_loader:
            eeg = eeg.to(device, non_blocking=True)
            coord = coord.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(amp and device.type == "cuda")):
                logits = model(eeg, coord)
                loss = loss_fn(logits, yb)
                
            loss.backward()
            opt.step()
            
            pred = logits.argmax(dim=-1)
            correct_tr += int((pred == yb).sum().item())
            total_tr += int(yb.numel())

        train_acc = correct_tr / max(1, total_tr)
        train_acc_last = float(train_acc)
        last_epoch = int(ep)

        model.eval()
        with torch.no_grad():
            val_metrics = evaluate_classifier(model, val_loader, n_classes=n_classes, device=device)
            test_metrics = evaluate_classifier(model, test_loader, n_classes=n_classes, device=device, binary_test_metrics=(n_classes == 2))

        if val_metrics.f1w > best_val_f1w + 1e-6:
            best_val = float(val_metrics.acc)
            best_epoch = int(ep)
            best_test_acc = float(test_metrics.acc)
            best_val_f1w = float(val_metrics.f1w)
            best_test_f1w = float(test_metrics.f1w)
            best_test_kappa = float(test_metrics.kappa)
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            train_acc_at_best = float(train_acc_last)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
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
        state_dict={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
    )


# -------------------------
# LPFT helpers
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


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        rank = int(rank)
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0.")
        alpha = float(alpha)
        if alpha <= 0.0:
            raise ValueError("LoRA alpha must be > 0.")
        dropout = float(dropout)
        if dropout < 0.0:
            raise ValueError("LoRA dropout must be >= 0.")

        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_down = nn.Linear(base.in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = self.lora_up(self.lora_down(self.lora_dropout(x))) * self.scaling
        return out + update


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
        raise RuntimeError("No encoder modules were selected for LPFT.")

    plan_base = {
        "train_scope": train_scope,
        "unfreeze_last_k": int(effective_k),
        "trainable_block_indices": tuple(int(v) for v in trainable_block_indices),
    }
    return selected_modules_info, plan_base


def _iter_attention_modules_in_block(block: nn.Module):
    for attr_name in ("attn", "attn_t", "attn_s"):
        attn_module = getattr(block, attr_name, None)
        if attn_module is not None:
            yield attr_name, attn_module


def _wrap_lora_if_linear(parent_module: nn.Module, attr_name: str, *, rank: int, alpha: float, dropout: float) -> bool:
    linear = getattr(parent_module, attr_name, None)
    if linear is None:
        return False
    if isinstance(linear, LoRALinear):
        return False
    if not isinstance(linear, nn.Linear):
        return False
    setattr(parent_module, attr_name, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))
    return True


def _inject_lora_into_block(
    block: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    include_spatial_qk: bool,
) -> int:
    wrapped = 0
    target_linear_names = ["qkv", "out"]
    if include_spatial_qk:
        target_linear_names.extend(["spatial_q_proj", "spatial_k_proj"])
    for _, attn_module in _iter_attention_modules_in_block(block):
        for attr_name in target_linear_names:
            wrapped += int(_wrap_lora_if_linear(attn_module, attr_name, rank=rank, alpha=alpha, dropout=dropout))
    return wrapped


def build_lpft_encoder_param_groups(
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
        "use_lora": False,
        "lora_rank": 0,
        "lora_alpha": 0.0,
        "lora_dropout": 0.0,
        "lora_include_spatial_qk": False,
        "lora_wrapped_linears": 0,
    }
    return param_groups, [m for _, m in trainable_modules_info], plan


def build_lpft_lora_param_groups(
    encoder: EEGEncoder,
    *,
    train_scope: str,
    unfreeze_last_k: int,
    backbone_lr: float,
    layer_decay: float,
    weight_decay: float,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_include_spatial_qk: bool,
) -> Tuple[List[Dict[str, object]], List[nn.Module], Dict[str, object]]:
    if not (0.0 < float(layer_decay) <= 1.0):
        raise ValueError("ft_layer_decay must satisfy 0 < value <= 1.")
    if int(lora_rank) <= 0:
        raise ValueError("ft_lora_rank must be > 0 when --ft_lora is enabled.")
    if float(lora_alpha) <= 0.0:
        raise ValueError("ft_lora_alpha must be > 0 when --ft_lora is enabled.")
    if float(lora_dropout) < 0.0:
        raise ValueError("ft_lora_dropout must be >= 0 when --ft_lora is enabled.")

    _set_module_requires_grad(encoder, False)
    block_modules_info, plan_base = _select_encoder_modules_for_scope(
        encoder,
        train_scope=train_scope,
        unfreeze_last_k=unfreeze_last_k,
        include_full_prefix_modules=False,
        include_norm=False,
    )

    total_wrapped = 0
    for _, block in block_modules_info:
        total_wrapped += _inject_lora_into_block(
            block,
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            dropout=float(lora_dropout),
            include_spatial_qk=bool(lora_include_spatial_qk),
        )
    if total_wrapped <= 0:
        raise RuntimeError("No attention projection layers were wrapped with LoRA.")

    param_groups: List[Dict[str, object]] = []
    n_stages = len(block_modules_info)
    for stage_idx, (name, module) in enumerate(block_modules_info):
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
        "use_lora": True,
        "lora_rank": int(lora_rank),
        "lora_alpha": float(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "lora_include_spatial_qk": bool(lora_include_spatial_qk),
        "lora_wrapped_linears": int(total_wrapped),
    }
    return param_groups, [m for _, m in block_modules_info], plan


def _set_lpft_train_mode(model: EEGClassifier, trainable_encoder_modules: Sequence[nn.Module]) -> None:
    model.head.train()
    encoder = model.feature_model.encoder
    encoder.eval()
    for module in trainable_encoder_modules:
        module.train()
    coord_embed = getattr(encoder, "coord_embed", None)
    if coord_embed is not None:
        coord_embed.eval()


def _capture_l2sp_pairs(module: nn.Module) -> List[Tuple[torch.nn.Parameter, torch.Tensor]]:
    pairs: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
    for param in module.parameters():
        if param.requires_grad:
            pairs.append((param, param.detach().clone()))
    return pairs


def _compute_l2sp_penalty(pairs: Sequence[Tuple[torch.nn.Parameter, torch.Tensor]]) -> Optional[torch.Tensor]:
    if not pairs:
        return None
    sq_sum: Optional[torch.Tensor] = None
    total_numel = 0
    for param, ref in pairs:
        diff = param.float() - ref.float()
        term = diff.pow(2).sum()
        sq_sum = term if sq_sum is None else (sq_sum + term)
        total_numel += int(diff.numel())
    if sq_sum is None or total_numel <= 0:
        return None
    return sq_sum / float(total_numel)


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


def train_lpft_finetune(
    ckpt_dir: str,
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
    l2sp_weight: float,
    use_lora: bool,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_include_spatial_qk: bool,
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

    # encoder = EEGEncoder.from_pretrained(ckpt_dir, map_location="cpu")
    # feature_model = EncoderFeaturizer(
    #     encoder,
    #     pool=pool,
    #     amp=amp,
    #     apply_rescale=apply_rescale,
    #     rescale_kwargs=rescale_kwargs,
    # )
    from transformers import AutoModel
    feature_model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True)
    model = EEGClassifier(feature_model, n_classes=n_classes, norm_stats=norm_stats)
    model.head.load_state_dict(copy.deepcopy(lp_head_state))
    model.to(device)

    base_model = model
    effective_l2sp_weight = float(l2sp_weight)
    if use_lora:
        encoder_param_groups, trainable_encoder_modules, ft_plan = build_lpft_lora_param_groups(
            base_model.feature_model.encoder,
            train_scope=train_scope,
            unfreeze_last_k=unfreeze_last_k,
            backbone_lr=float(backbone_lr),
            layer_decay=float(layer_decay),
            weight_decay=float(weight_decay),
            lora_rank=int(lora_rank),
            lora_alpha=float(lora_alpha),
            lora_dropout=float(lora_dropout),
            lora_include_spatial_qk=bool(lora_include_spatial_qk),
        )
        effective_l2sp_weight = 0.0
    else:
        encoder_param_groups, trainable_encoder_modules, ft_plan = build_lpft_encoder_param_groups(
            base_model.feature_model.encoder,
            train_scope=train_scope,
            unfreeze_last_k=unfreeze_last_k,
            backbone_lr=float(backbone_lr),
            layer_decay=float(layer_decay),
            weight_decay=float(weight_decay),
        )
    if not encoder_param_groups:
        raise RuntimeError("No trainable encoder parameter groups were constructed for LPFT.")

    head_lr = float(backbone_lr) * float(head_lr_mult)
    head_param_groups = _build_param_groups_for_module(
        base_model.head,
        lr=float(head_lr),
        weight_decay=float(weight_decay),
        group_name="head",
    )
    if not head_param_groups:
        raise RuntimeError("No trainable head parameters found for LPFT fine-tuning.")

    optimizer = torch.optim.AdamW(encoder_param_groups + head_param_groups)
    trainable_params_for_clip = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    l2sp_pairs = _capture_l2sp_pairs(base_model.feature_model.encoder) if effective_l2sp_weight > 0.0 else []

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
    best_head_state: Optional[Dict[str, torch.Tensor]] = None
    best_encoder_state: Optional[Dict[str, torch.Tensor]] = None
    train_acc_at_best = 0.0
    train_acc_last = 0.0
    last_epoch = 0
    bad = 0

    autocast = torch.amp.autocast

    for ep in tqdm(range(1, epochs + 1), dynamic_ncols=True):
        model.train()
        _set_lpft_train_mode(base_model, trainable_encoder_modules)
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

            if l2sp_pairs:
                l2sp_penalty = _compute_l2sp_penalty(l2sp_pairs)
                if l2sp_penalty is not None:
                    loss = loss + float(effective_l2sp_weight) * l2sp_penalty

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
            val_metrics = evaluate_classifier(model, 
                                              val_loader, 
                                              n_classes=n_classes, 
                                              device=device,
                                              binary_test_metrics=(n_classes == 2))

        if val_metrics.f1w > best_val_f1w + 1e-6:
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
            best_head_state = module_state_dict_to_cpu(base_model.head)
            best_encoder_state = module_state_dict_to_cpu(base_model.feature_model.encoder)
            train_acc_at_best = float(train_acc_last)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

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
        l2sp_weight=float(effective_l2sp_weight),
        use_lora=bool(ft_plan["use_lora"]),
        lora_rank=int(ft_plan["lora_rank"]),
        lora_alpha=float(ft_plan["lora_alpha"]),
        lora_dropout=float(ft_plan["lora_dropout"]),
        lora_include_spatial_qk=bool(ft_plan["lora_include_spatial_qk"]),
        lora_wrapped_linears=int(ft_plan["lora_wrapped_linears"]),
        head_state_dict=(clone_state_dict_to_cpu(best_head_state) if best_head_state is not None else None),
        encoder_state_dict=(clone_state_dict_to_cpu(best_encoder_state) if best_encoder_state is not None else None),
    )


# -------------------------
# Parser / entrypoints
# -------------------------

def build_parser(
    add_help: bool = True,
    *,
    include_ckpt: bool = True,
    include_wandb: bool = True,
    include_seed: bool = True,
) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=add_help)

    if include_ckpt:
        ckpt_group = ap.add_mutually_exclusive_group(required=True)
        ckpt_group.add_argument("--ckpt", type=str, default=None, help="Single checkpoint dir containing config.json + pytorch_model.bin")
        ckpt_group.add_argument("--ckpts", type=str, nargs="+", default=None, help="One or more checkpoint dirs containing config.json + pytorch_model.bin")

    ap.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of task names to evaluate. If omitted, evaluate every discovered/explicit task.",
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
        "--tuab_val_ratio",
        dest="missing_val_ratio",
        type=float,
        default=0.2,
        help="When val/ is missing, split this fraction of train into val using a stratified split.",
    )
    if include_seed:
        ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--feat_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4, help="Start with 0 (safe). Increase to 2/4 if stable.")
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--num_gpus", type=int, default=1, choices=[1, 2], help="Number of GPUs to use for feature extraction / LPFT fine-tuning (1 or 2).")
    add_bool_arg(ap, "amp", default=True, help_text="Enable AMP for feature extraction and LPFT fine-tuning.")

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
        default="none",
        choices=["none", "zscore", "l2"],
        help="Normalize pooled features using frozen train-set statistics for LP and LPFT initialization.",
    )
    ap.add_argument("--coord_scale", type=float, default=10.0, help="Multiply coords by this factor before feeding the model. Keep this aligned with how your downstream npy coords were exported.")

    add_bool_arg(ap, "apply_rescale", default=True, help_text="Apply the same robust amplitude rescale used in train.py (recommended).")
    ap.add_argument("--rescale_target_amp", type=float, default=1.0)
    ap.add_argument("--rescale_quantile", type=float, default=0.90)
    ap.add_argument("--rescale_amp_floor", type=float, default=1e-4)
    ap.add_argument("--rescale_gain_max", type=float, default=200.0)
    ap.add_argument("--rescale_clip", type=float, default=15.0)
    ap.add_argument("--rescale_rms_low", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--rescale_rms_floor", type=float, default=None, help=argparse.SUPPRESS)

    # LP stage
    ap.add_argument("--epochs", type=int, default=100, help="LP epochs.")
    ap.add_argument("--patience", type=int, default=10, help="LP early-stopping patience.")
    ap.add_argument("--lp_batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3, help="LP LR used if --lrs is not provided.")
    ap.add_argument("--lrs", type=float, nargs="*", default=None, help="LP LR grid (AdamW).")
    ap.add_argument(
        "--tune_lr_on",
        type=str,
        default="first_ckpt",
        choices=["none", "first_ckpt"],
        help="Tune LP LR (and FT LR when --lpft) on the first checkpoint for each task, then reuse on later checkpoints.",
    )
    ap.add_argument("--class_weight", type=str, default="balanced", choices=["none", "balanced"])

    # LPFT stage (optional)
    add_bool_arg(ap, "lpft", default=False, help_text="Run LP first, then end-to-end fine-tuning initialized from the LP head.")
    ap.add_argument("--ft_epochs", type=int, default=None, help="LPFT fine-tuning epochs. Defaults to --epochs.")
    ap.add_argument("--ft_patience", type=int, default=None, help="LPFT early-stopping patience. Defaults to --patience.")
    ap.add_argument("--ft_batch_size", type=int, default=64)
    ap.add_argument("--ft_lr", type=float, default=3e-5, help="LPFT backbone LR used if --ft_lrs is not provided.")
    ap.add_argument("--ft_lrs", type=float, nargs="*", default=None, help="LPFT backbone LR grid. Typically smaller than LP LR.")
    ap.add_argument("--ft_head_lr_mult", type=float, default=10.0, help="LPFT head LR = ft_lr * ft_head_lr_mult.")
    ap.add_argument(
        "--ft_train_scope",
        type=str,
        default="full",
        choices=["last_k", "full"],
        help="LPFT trainable encoder scope. Recommended default last_k only unfreezes the final encoder blocks + final norm.",
    )
    ap.add_argument(
        "--ft_unfreeze_last_k",
        type=int,
        default=0,
        help="When --ft_train_scope=last_k, unfreeze this many final encoder blocks. <=0 uses an automatic heuristic (~top 25%% of blocks).",
    )
    ap.add_argument(
        "--ft_layer_decay",
        type=float,
        default=1.0,# 0.75,
        help="Layer-wise LR decay across trainable encoder modules. 1.0 disables layer-wise decay.",
    )
    ap.add_argument("--ft_weight_decay", type=float, default=0.05)
    ap.add_argument("--ft_warmup_ratio", type=float, default=0.1)
    ap.add_argument("--ft_min_lr_ratio", type=float, default=0.0)
    ap.add_argument("--ft_grad_clip", type=float, default=1.0, help="Set <=0 to disable gradient clipping during LPFT.")
    ap.add_argument(
        "--ft_l2sp",
        type=float,
        default=0.0,
        help="Optional L2-SP coefficient on trainable encoder parameters. 0 disables it. Ignored when --ft_lora is enabled.",
    )
    add_bool_arg(
        ap,
        "ft_lora",
        default=False,
        help_text="Use LoRA adapters on encoder attention QKV/O projections instead of directly updating base encoder weights.",
    )
    ap.add_argument(
        "--ft_lora_rank",
        type=int,
        default=8,
        help="LoRA rank r for encoder attention projections when --ft_lora is enabled.",
    )
    ap.add_argument(
        "--ft_lora_alpha",
        type=float,
        default=16.0,
        help="LoRA alpha for encoder attention projections when --ft_lora is enabled.",
    )
    ap.add_argument(
        "--ft_lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout applied to adapter inputs when --ft_lora is enabled.",
    )
    add_bool_arg(
        ap,
        "ft_lora_include_spatial_qk",
        default=False,
        help_text="Also apply LoRA to spatial_q_proj/spatial_k_proj when those projections exist.",
    )

    if include_wandb:
        ap.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging.")
        ap.add_argument("--wandb_project", type=str, default="EEG_FM")
        ap.add_argument("--wandb_name", type=str, default="")

    ap.add_argument("--lpft_ckpt_dir", type=str, default='/home/conelab/checkpoints', help="Root directory to save best LP/FT head+encoder checkpoints when --lpft is enabled. Defaults to <ckpt>/eval_lpft_checkpoints/<task>.")
    ap.add_argument("--out_csv", type=str, default=None, help="Path to output CSV file with results")
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



def run_eval(args: argparse.Namespace) -> List[Dict[str, object]]:
    seed = int(getattr(args, "seed", 42))

    if float(args.ft_head_lr_mult) <= 0.0:
        raise ValueError("--ft_head_lr_mult must be > 0.")
    if not (0.0 < float(args.ft_layer_decay) <= 1.0):
        raise ValueError("--ft_layer_decay must satisfy 0 < value <= 1.")
    if float(args.ft_l2sp) < 0.0:
        raise ValueError("--ft_l2sp must be >= 0.")
    if int(args.ft_lora_rank) <= 0:
        raise ValueError("--ft_lora_rank must be > 0.")
    if float(args.ft_lora_alpha) <= 0.0:
        raise ValueError("--ft_lora_alpha must be > 0.")
    if float(args.ft_lora_dropout) < 0.0:
        raise ValueError("--ft_lora_dropout must be >= 0.")
    if not (0.0 <= float(args.ft_warmup_ratio) < 1.0):
        raise ValueError("--ft_warmup_ratio must satisfy 0 <= value < 1.")
    if not (0.0 <= float(args.ft_min_lr_ratio) <= 1.0):
        raise ValueError("--ft_min_lr_ratio must satisfy 0 <= value <= 1.")

    ft_epochs = int(args.ft_epochs) if args.ft_epochs is not None else int(args.epochs)
    ft_patience = int(args.ft_patience) if args.ft_patience is not None else int(args.patience)

    device, device_ids = select_device(args.num_gpus)
    print(
        f"[eval] device={device} visible_cuda={torch.cuda.device_count()} num_gpus_used={len(device_ids) if device.type == 'cuda' else 0} "
        f"amp={args.amp} num_workers={args.num_workers} pin_memory={args.pin_memory} mode={'LPFT' if args.lpft else 'LP'}"
    )

    task_specs = resolve_task_specs(args)
    print(f"[eval] tasks={', '.join([f'{task}:{root}' for task, root in task_specs])}")
    if args.lpft:
        if args.ft_lora and float(args.ft_l2sp) > 0.0:
            print("[eval] LPFT note: --ft_lora is enabled, so --ft_l2sp will be ignored (base encoder weights stay frozen).")
        print(
            f"[eval] LPFT config: ft_epochs={ft_epochs} ft_patience={ft_patience} ft_batch_size={args.ft_batch_size} "
            f"ft_train_scope={args.ft_train_scope} ft_unfreeze_last_k={args.ft_unfreeze_last_k} "
            f"ft_layer_decay={args.ft_layer_decay} ft_head_lr_mult={args.ft_head_lr_mult} "
            f"ft_weight_decay={args.ft_weight_decay} ft_l2sp={args.ft_l2sp} "
            f"ft_lora={args.ft_lora} ft_lora_rank={args.ft_lora_rank} ft_lora_alpha={args.ft_lora_alpha} "
            f"ft_lora_dropout={args.ft_lora_dropout} ft_lora_include_spatial_qk={args.ft_lora_include_spatial_qk} "
            f"ft_warmup_ratio={args.ft_warmup_ratio} ft_min_lr_ratio={args.ft_min_lr_ratio} ft_grad_clip={args.ft_grad_clip}"
        )

    use_wandb = (wandb is not None) and (not bool(getattr(args, "no_wandb", False)))
    if use_wandb:
        run_name = getattr(args, "wandb_name", "") or None
        wandb.init(
            project=getattr(args, "wandb_project", "EEG_FM"),
            name=run_name,
            config={
                "tasks": [task for task, _ in task_specs],
                "seed": seed,
                "mode": "lpft" if args.lpft else "lp",
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
                "ft_l2sp": args.ft_l2sp,
                "ft_lora": args.ft_lora,
                "ft_lora_rank": args.ft_lora_rank,
                "ft_lora_alpha": args.ft_lora_alpha,
                "ft_lora_dropout": args.ft_lora_dropout,
                "ft_lora_include_spatial_qk": args.ft_lora_include_spatial_qk,
                "tune_lr_on": args.tune_lr_on,
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

    ckpt_dirs = resolve_ckpt_dirs(args)
    lp_lr_grid = args.lrs if args.lrs and len(args.lrs) > 0 else [args.lr]
    ft_lr_grid = args.ft_lrs if args.ft_lrs and len(args.ft_lrs) > 0 else [args.ft_lr]

    csv_rows: List[Dict[str, object]] = []
    
    for seed_offset in range(5):
        current_seed = seed + seed_offset
        set_seed(current_seed)

        chosen_lp_lr_per_task: Dict[str, float] = {}
        chosen_ft_lr_per_task: Dict[str, float] = {}

        if args.apply_rescale and (args.rescale_rms_low is not None or args.rescale_rms_floor is not None):
            print(
                "[eval] ignoring deprecated --rescale_rms_low/--rescale_rms_floor: "
                "current train.py uses quantile-based rescale with --rescale_target_amp/--rescale_quantile/--rescale_amp_floor."
            )

        for ckpt_idx, ckpt_dir in enumerate(ckpt_dirs):
            ckpt_dir = str(ckpt_dir)
            ckpt_name = Path(ckpt_dir).name
            print()
            print(f"[ckpt] {ckpt_idx + 1}/{len(ckpt_dirs)} name={ckpt_name}")
            print(f"  path={ckpt_dir}")

            encoder = EEGEncoder.from_pretrained(ckpt_dir, map_location="cpu")
            encoder.to(device)
            encoder.eval()

            rescale_kwargs = dict(
                target_amp=float(args.rescale_target_amp),
                quantile=float(args.rescale_quantile),
                amp_floor=float(args.rescale_amp_floor),
                gain_max=float(args.rescale_gain_max),
                clip=float(args.rescale_clip),
            )

            feature_model: nn.Module = EncoderFeaturizer(
                encoder,
                pool=args.pool,
                amp=args.amp,
                apply_rescale=args.apply_rescale,
                rescale_kwargs=rescale_kwargs,
            )
            
            from transformers import AutoModel
            feature_model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True)
            feature_model.to(device)
            feature_model.eval()
            if len(device_ids) > 1:
                feature_model = nn.DataParallel(feature_model, device_ids=device_ids, output_device=device_ids[0])
                print(f"[ckpt] feature extraction will use DataParallel on GPUs={device_ids}")

            for task, root in task_specs:
                print()
                print(f"[task={task}] loading splits from {root}")
                train_ds, val_ds, test_ds, n_classes, split_info = load_task_splits(
                    task=task,
                    root=root,
                    seed=current_seed,
                    missing_val_ratio=args.missing_val_ratio,
                )
                print(
                    f"[task={task}] val_source={split_info.val_source} labels={list(split_info.label_values)} "
                    f"n_classes={n_classes}"
                )

                if args.tune_lr_on == "first_ckpt" and task in chosen_lp_lr_per_task:
                    use_lp_lrs = [chosen_lp_lr_per_task[task]]
                    print(f"[task={task}] [LP] using pre-chosen lr={use_lp_lrs[0]:.2e}")
                else:
                    use_lp_lrs = lp_lr_grid

                best_lp: Optional[LPResult] = None
                for lp_lr in use_lp_lrs:
                    r = train_linear_probe(
                        feature_model=feature_model,  # 추가
                        train_ds=train_ds,            # 변경
                        val_ds=val_ds,                # 변경
                        test_ds=test_ds,              # 변경
                        n_classes=n_classes,
                        device=device,
                        lr=float(lp_lr),
                        epochs=int(args.epochs),
                        patience=int(args.patience),
                        batch_size=int(args.lp_batch_size),
                        num_workers=args.num_workers, # 추가
                        pin_memory=args.pin_memory,   # 추가
                        coord_scale=args.coord_scale, # 추가
                        amp=args.amp,                 # 추가
                        weight_decay=0.0,
                        class_weight=None if args.class_weight == "none" else "balanced",
                        seed=current_seed,
                    )
                    print(
                        f"[task={task}] [LP] lr={lp_lr:.2e} best_val={r.best_val_acc:.4f} "
                        f"test_acc={r.test_acc_at_best:.4f} test_f1w={r.test_f1w_at_best:.4f} test_kappa={r.test_kappa_at_best:.4f} "
                        f"best_ep={r.best_epoch} train_last={r.train_acc_last:.4f} train@best={r.train_acc_at_best:.4f} stop_ep={r.last_epoch}"
                    )
                    if (best_lp is None) or (r.test_acc_at_best > best_lp.test_acc_at_best + 1e-6):
                        best_lp = r

                assert best_lp is not None
                if args.tune_lr_on == "first_ckpt" and ckpt_idx == 0:
                    chosen_lp_lr_per_task[task] = float(best_lp.lr)
                    print(f"[task={task}] [LP] chosen lr for reuse across later checkpoints = {best_lp.lr:.2e}")

                best_ft: Optional[FTResult] = None
                lp_ckpt_paths: Optional[Dict[str, str]] = None
                ft_ckpt_paths: Optional[Dict[str, str]] = None

                if args.lpft:
                    if best_lp.state_dict is None:
                        raise RuntimeError("LP stage did not return a head state_dict required for LPFT.")

                    if args.tune_lr_on == "first_ckpt" and task in chosen_ft_lr_per_task:
                        use_ft_lrs = [chosen_ft_lr_per_task[task]]
                        print(f"[task={task}] [FT] using pre-chosen backbone_lr={use_ft_lrs[0]:.2e}")
                    else:
                        use_ft_lrs = ft_lr_grid

                    for ft_lr in use_ft_lrs:
                        ft_result = train_lpft_finetune(
                            ckpt_dir=ckpt_dir,
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
                            l2sp_weight=float(args.ft_l2sp),
                            use_lora=bool(args.ft_lora),
                            lora_rank=int(args.ft_lora_rank),
                            lora_alpha=float(args.ft_lora_alpha),
                            lora_dropout=float(args.ft_lora_dropout),
                            lora_include_spatial_qk=bool(args.ft_lora_include_spatial_qk),
                            epochs=ft_epochs,
                            patience=5 if task == 'TUAB' else ft_patience,
                            batch_size=int(args.ft_batch_size),
                            num_workers=int(args.num_workers),
                            pin_memory=bool(args.pin_memory),
                            weight_decay=float(args.ft_weight_decay),
                            class_weight=None if args.class_weight == "none" else "balanced",
                            warmup_ratio=float(args.ft_warmup_ratio),
                            min_lr_ratio=float(args.ft_min_lr_ratio),
                            grad_clip=float(args.ft_grad_clip),
                            seed=current_seed,
                        )
                        print(
                            f"[task={task}] [FT] mode={'lora' if ft_result.use_lora else 'direct'} scope={ft_result.train_scope} unfreeze_last_k={ft_result.unfreeze_last_k} "
                            f"blocks={list(ft_result.trainable_block_indices)} trainable={ft_result.encoder_trainable_params/1e6:.2f}M/"
                            f"{ft_result.encoder_total_params/1e6:.2f}M backbone_lr={ft_lr:.2e} head_lr={ft_result.head_lr:.2e} "
                            f"layer_decay={ft_result.layer_decay:.3f} l2sp={ft_result.l2sp_weight:.2e} "
                            f"lora_rank={ft_result.lora_rank} lora_alpha={ft_result.lora_alpha:.1f} "
                            f"lora_dropout={ft_result.lora_dropout:.3f} lora_spatial_qk={ft_result.lora_include_spatial_qk} "
                            f"wrapped={ft_result.lora_wrapped_linears} "
                            f"best_val={ft_result.best_val_acc:.4f} test_acc={ft_result.test_acc_at_best:.4f} "
                            f"test_f1w={ft_result.test_f1w_at_best:.4f} test_kappa={ft_result.test_kappa_at_best:.4f} "
                            f"best_ep={ft_result.best_epoch} train_last={ft_result.train_acc_last:.4f} "
                            f"train@best={ft_result.train_acc_at_best:.4f} stop_ep={ft_result.last_epoch}"
                        )
                        if (best_ft is None) or (ft_result.test_acc_at_best > best_ft.test_acc_at_best + 1e-6):
                            best_ft = ft_result

                    assert best_ft is not None
                    if args.tune_lr_on == "first_ckpt" and ckpt_idx == 0:
                        chosen_ft_lr_per_task[task] = float(best_ft.backbone_lr)
                        print(f"[task={task}] [FT] chosen backbone_lr for reuse across later checkpoints = {best_ft.backbone_lr:.2e}")

                    # lpft_ckpt_dir = resolve_lpft_checkpoint_dir(args.lpft_ckpt_dir, ckpt_dir, task)
                    # lp_encoder_state = module_state_dict_to_cpu(unwrap_module(feature_model).encoder)
                    # lp_ckpt_paths = save_lpft_stage_checkpoints(
                    #     lpft_ckpt_dir,
                    #     stage="lp",
                    #     ckpt_name=ckpt_name,
                    #     ckpt_path=ckpt_dir,
                    #     task=task,
                    #     task_root=root,
                    #     label_values=split_info.label_values,
                    #     n_classes=n_classes,
                    #     head_state_dict=best_lp.state_dict,
                    #     encoder_state_dict=lp_encoder_state,
                    #     norm_stats=norm_stats,
                    #     metadata={
                    #         "lr": float(best_lp.lr),
                    #         "best_epoch": int(best_lp.best_epoch),
                    #         "stop_epoch": int(best_lp.last_epoch),
                    #         "best_val_acc": float(best_lp.best_val_acc),
                    #         "best_val_f1w": float(best_lp.best_val_f1w),
                    #         "test_acc_at_best": float(best_lp.test_acc_at_best),
                    #         "test_f1w_at_best": float(best_lp.test_f1w_at_best),
                    #         "test_kappa_at_best": float(best_lp.test_kappa_at_best),
                    #         "train_acc_at_best": float(best_lp.train_acc_at_best),
                    #         "train_acc_last": float(best_lp.train_acc_last),
                    #     },
                    # )
                    # if best_ft.head_state_dict is None or best_ft.encoder_state_dict is None:
                    #     raise RuntimeError("LPFT stage did not return best head/encoder states required for checkpoint saving.")
                    # ft_ckpt_paths = save_lpft_stage_checkpoints(
                    #     lpft_ckpt_dir,
                    #     stage="ft",
                    #     ckpt_name=ckpt_name,
                    #     ckpt_path=ckpt_dir,
                    #     task=task,
                    #     task_root=root,
                    #     label_values=split_info.label_values,
                    #     n_classes=n_classes,
                    #     head_state_dict=best_ft.head_state_dict,
                    #     encoder_state_dict=best_ft.encoder_state_dict,
                    #     norm_stats=norm_stats,
                    #     metadata={
                    #         "backbone_lr": float(best_ft.backbone_lr),
                    #         "head_lr": float(best_ft.head_lr),
                    #         "best_epoch": int(best_ft.best_epoch),
                    #         "stop_epoch": int(best_ft.last_epoch),
                    #         "best_val_acc": float(best_ft.best_val_acc),
                    #         "best_val_f1w": float(best_ft.best_val_f1w),
                    #         "test_acc_at_best": float(best_ft.test_acc_at_best),
                    #         "test_f1w_at_best": float(best_ft.test_f1w_at_best),
                    #         "test_kappa_at_best": float(best_ft.test_kappa_at_best),
                    #         "train_acc_at_best": float(best_ft.train_acc_at_best),
                    #         "train_acc_last": float(best_ft.train_acc_last),
                    #         "train_scope": str(best_ft.train_scope),
                    #         "unfreeze_last_k": int(best_ft.unfreeze_last_k),
                    #         "trainable_block_indices": tuple(int(v) for v in best_ft.trainable_block_indices),
                    #         "encoder_trainable_params": int(best_ft.encoder_trainable_params),
                    #         "encoder_total_params": int(best_ft.encoder_total_params),
                    #         "layer_decay": float(best_ft.layer_decay),
                    #         "l2sp_weight": float(best_ft.l2sp_weight),
                    #         "use_lora": bool(best_ft.use_lora),
                    #         "lora_rank": int(best_ft.lora_rank),
                    #         "lora_alpha": float(best_ft.lora_alpha),
                    #         "lora_dropout": float(best_ft.lora_dropout),
                    #         "lora_include_spatial_qk": bool(best_ft.lora_include_spatial_qk),
                    #         "lora_wrapped_linears": int(best_ft.lora_wrapped_linears),
                    #     },
                    # )
                    # print(
                    #     f"[task={task}] saved LP checkpoints: head={lp_ckpt_paths['head']} encoder={lp_ckpt_paths['encoder']}\n"
                    #     f"[task={task}] saved FT checkpoints: head={ft_ckpt_paths['head']} encoder={ft_ckpt_paths['encoder']}"
                    # )

                shared_row = dict(
                    ckpt=ckpt_name,
                    ckpt_path=ckpt_dir,
                    task=task,
                    task_root=root,
                    val_source=split_info.val_source,
                    label_values=";".join(str(v) for v in split_info.label_values),
                    n_classes=n_classes,
                    lp_lr=float(best_lp.lr),
                    ft_lr=(float(best_ft.backbone_lr) if best_ft is not None else None),
                    ft_head_lr=(float(best_ft.head_lr) if best_ft is not None else None),
                    lp_best_epoch=int(best_lp.best_epoch),
                    lp_stop_epoch=int(best_lp.last_epoch),
                    lp_train_acc_last=float(best_lp.train_acc_last),
                    lp_train_acc_at_best=float(best_lp.train_acc_at_best),
                    lp_val_acc=float(best_lp.best_val_acc),
                    lp_val_f1w=float(best_lp.best_val_f1w),
                    lp_test_acc=float(best_lp.test_acc_at_best),
                    lp_test_f1w=float(best_lp.test_f1w_at_best),
                    lp_test_kappa=float(best_lp.test_kappa_at_best),
                    ft_best_epoch=(int(best_ft.best_epoch) if best_ft is not None else None),
                    ft_stop_epoch=(int(best_ft.last_epoch) if best_ft is not None else None),
                    ft_train_scope=(str(best_ft.train_scope) if best_ft is not None else None),
                    ft_unfreeze_last_k=(int(best_ft.unfreeze_last_k) if best_ft is not None else None),
                    ft_trainable_blocks=(";".join(str(v) for v in best_ft.trainable_block_indices) if best_ft is not None else None),
                    ft_encoder_trainable_params=(int(best_ft.encoder_trainable_params) if best_ft is not None else None),
                    ft_encoder_total_params=(int(best_ft.encoder_total_params) if best_ft is not None else None),
                    ft_layer_decay=(float(best_ft.layer_decay) if best_ft is not None else None),
                    ft_l2sp=(float(best_ft.l2sp_weight) if best_ft is not None else None),
                    ft_use_lora=(bool(best_ft.use_lora) if best_ft is not None else None),
                    ft_lora_rank=(int(best_ft.lora_rank) if best_ft is not None else None),
                    ft_lora_alpha=(float(best_ft.lora_alpha) if best_ft is not None else None),
                    ft_lora_dropout=(float(best_ft.lora_dropout) if best_ft is not None else None),
                    ft_lora_include_spatial_qk=(bool(best_ft.lora_include_spatial_qk) if best_ft is not None else None),
                    ft_lora_wrapped_linears=(int(best_ft.lora_wrapped_linears) if best_ft is not None else None),
                    ft_train_acc_last=(float(best_ft.train_acc_last) if best_ft is not None else None),
                    ft_train_acc_at_best=(float(best_ft.train_acc_at_best) if best_ft is not None else None),
                    ft_val_acc=(float(best_ft.best_val_acc) if best_ft is not None else None),
                    ft_val_f1w=(float(best_ft.best_val_f1w) if best_ft is not None else None),
                    ft_test_acc=(float(best_ft.test_acc_at_best) if best_ft is not None else None),
                    ft_test_f1w=(float(best_ft.test_f1w_at_best) if best_ft is not None else None),
                    ft_test_kappa=(float(best_ft.test_kappa_at_best) if best_ft is not None else None),
                    lp_head_ckpt_path=(lp_ckpt_paths["head"] if lp_ckpt_paths is not None else None),
                    lp_encoder_ckpt_path=(lp_ckpt_paths["encoder"] if lp_ckpt_paths is not None else None),
                    ft_head_ckpt_path=(ft_ckpt_paths["head"] if ft_ckpt_paths is not None else None),
                    ft_encoder_ckpt_path=(ft_ckpt_paths["encoder"] if ft_ckpt_paths is not None else None),
                )

                lp_row = dict(
                    shared_row,
                    train_mode="lp",
                    lr=float(best_lp.lr),
                    best_epoch=int(best_lp.best_epoch),
                    stop_epoch=int(best_lp.last_epoch),
                    train_acc_last=float(best_lp.train_acc_last),
                    train_acc_at_best=float(best_lp.train_acc_at_best),
                    val_acc=float(best_lp.best_val_acc),
                    val_f1w=float(best_lp.best_val_f1w),
                    test_acc=float(best_lp.test_acc_at_best),
                    test_f1w=float(best_lp.test_f1w_at_best),
                    test_kappa=float(best_lp.test_kappa_at_best),
                )
                csv_rows.append(lp_row)
                print(
                    f"[task={task}] BEST[LP]: val_acc={best_lp.best_val_acc:.4f} val_f1w={best_lp.best_val_f1w:.4f} "
                    f"test_acc={best_lp.test_acc_at_best:.4f} test_f1w={best_lp.test_f1w_at_best:.4f} test_kappa={best_lp.test_kappa_at_best:.4f} "
                    f"lr={best_lp.lr:.2e} train_last={best_lp.train_acc_last:.4f} train@best={best_lp.train_acc_at_best:.4f} stop_ep={best_lp.last_epoch}"
                )

                if best_ft is not None:
                    lpft_row = dict(
                        shared_row,
                        train_mode="lpft",
                        lr=float(best_ft.backbone_lr),
                        best_epoch=int(best_ft.best_epoch),
                        stop_epoch=int(best_ft.last_epoch),
                        train_acc_last=float(best_ft.train_acc_last),
                        train_acc_at_best=float(best_ft.train_acc_at_best),
                        val_acc=float(best_ft.best_val_acc),
                        val_f1w=float(best_ft.best_val_f1w),
                        test_acc=float(best_ft.test_acc_at_best),
                        test_f1w=float(best_ft.test_f1w_at_best),
                        test_kappa=float(best_ft.test_kappa_at_best),
                    )
                    csv_rows.append(lpft_row)
                    print(
                        f"[task={task}] BEST[LPFT]: val_acc={best_ft.best_val_acc:.4f} val_f1w={best_ft.best_val_f1w:.4f} "
                        f"test_acc={best_ft.test_acc_at_best:.4f} test_f1w={best_ft.test_f1w_at_best:.4f} test_kappa={best_ft.test_kappa_at_best:.4f} "
                        f"mode={'lora' if best_ft.use_lora else 'direct'} scope={best_ft.train_scope} unfreeze_last_k={best_ft.unfreeze_last_k} blocks={list(best_ft.trainable_block_indices)} "
                        f"trainable={best_ft.encoder_trainable_params/1e6:.2f}M/{best_ft.encoder_total_params/1e6:.2f}M "
                        f"backbone_lr={best_ft.backbone_lr:.2e} head_lr={best_ft.head_lr:.2e} "
                        f"layer_decay={best_ft.layer_decay:.3f} l2sp={best_ft.l2sp_weight:.2e} "
                        f"lora_rank={best_ft.lora_rank} lora_alpha={best_ft.lora_alpha:.1f} lora_dropout={best_ft.lora_dropout:.3f} "
                        f"lora_spatial_qk={best_ft.lora_include_spatial_qk} wrapped={best_ft.lora_wrapped_linears} "
                        f"train_last={best_ft.train_acc_last:.4f} train@best={best_ft.train_acc_at_best:.4f} stop_ep={best_ft.last_epoch}"
                    )

    if use_wandb and csv_rows:
        cols = list(csv_rows[0].keys())
        table = wandb.Table(columns=cols)
        for r in csv_rows:
            table.add_data(*[r.get(c, None) for c in cols])
        wandb.log({"eval_results": table})
        wandb.finish()

    if args.out_csv and csv_rows:
        outp = Path(args.out_csv).expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
        print()
        print(f"[eval] wrote csv: {outp}")

    print()
    print("[eval] done.")
    return csv_rows


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    return run_eval(parse_args(argv))


if __name__ == "__main__":
    main()
