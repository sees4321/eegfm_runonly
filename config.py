
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import json
import os
from typing import Any, Dict, Optional, Tuple


@dataclass
class EEGModelConfig:
    # Tokenization
    sample_rate: int = 200
    patch_seconds: float = 1.0
    patch_hop_seconds: float = 1.0
    max_tokens: int = 4096

    # Core architecture (kept aligned with the validated 45M recipe)
    mlp_type: str = "swiglu"
    norm_type: str = "layernorm"
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0

    rope_theta: float = 10000.0
    rotary_pct: float = 1.0

    encoder_arch: str = "hybrid"
    full_attn_every: int = 4
    full_attn_use_spatial_bias: bool = False

    spatial_bias_type: str = "legendre"
    spatial_bias_degree: int = 8
    spatial_bias_use_unit_sphere: bool = True
    spatial_bias_scale: float = 1.0

    coord_jitter_std: float = 0.05
    coord_jitter_prob: float = 0.5
    coord_w_init: float = 0.0

    spatial_qk_type: str = "legendre_anchor"
    spatial_qk_num_anchors: int = 32
    spatial_qk_degree: int = 8
    spatial_qk_feat_dim: int = 64
    spatial_qk_scale: float = 1.0

    predictor_type: str = "cross_attn"
    predictor_layers: int = 2
    predictor_n_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    query_token_init_std: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(path: str) -> "EEGModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        allowed = {fd.name for fd in fields(EEGModelConfig)}
        d_f = {k: v for k, v in d.items() if k in allowed}
        return EEGModelConfig(**d_f)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


@dataclass
class TrainConfig:
    # Repro / runtime
    seed: int = 42
    torch_deterministic: bool = False
    cudnn_benchmark: bool = True
    mixed_precision: str = "bf16"

    # Data
    shards_txt: str = ""
    cache_dir: str = ""
    shard_shuffle: int = 0
    sample_shuffle: int = 256
    post_split_shuffle: int = 128
    cache_max_bytes: int = 0
    eviction_interval: int = 8

    tokens_per_batch: int = 16384  # per-rank valid-token target for ShapeBatcher
    max_samples_per_batch: int = 256
    num_workers: int = 4

    # Masking (run-only recipe keeps style-3 worker-side masks)
    mask_time_prob: float = 0.8
    mask_spatial_prob: float = 0.8
    mask_dilate_time: int = 0
    time_mask_style: int = 3
    time_mask_ratio_min: float = 0.15
    time_mask_ratio_max: float = 0.35
    spatial_mask_ratio_min: float = 0.10
    spatial_mask_ratio_max: float = 0.30

    # Student augmentations
    aug_gain_min: float = 0.8
    aug_gain_max: float = 1.2
    aug_channel_gain_std: float = 0.0
    aug_noise_std_min: float = 0.00
    aug_noise_std_max: float = 0.03
    aug_channel_drop_prob: float = 0.00

    # Optimization
    lr: float = 3.0e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # IMPORTANT: global effective target tokens per optimizer step (summed across ranks)
    tokens_per_update: int = 131072
    max_steps: int = 45850
    warmup_steps: int = 2000
    lr_cooldown_frac: float = 0.10
    min_lr: float = 0.0

    ema_momentum: float = 0.996
    ema_momentum_final: float = 0.9999

    # Relational spectral auxiliary loss (kept because the validated 45M config used it)
    spec_aux_mode: str = "relational"
    spec_rel_proj_dim: int = 128
    spec_rel_subsample_tokens: int = 512
    spec_rel_tau_z: float = 0.1
    spec_rel_tau_s: float = 0.1
    spec_rel_weight: float = 0.05
    spec_rel_warmup_steps: int = 1500
    spec_rel_ramp_steps: int = 3000
    spec_rel_decay_step: int = 16000
    spec_rel_final_weight: float = 0.005

    # Logging / checkpoints
    output_dir: str = "./checkpoints/eegfm"
    log_every: int = 50
    save_every: int = 5000
    save_every_pass: bool = True
    use_wandb: bool = True
    wandb_project: str = "EEG_FM"
    run_name: Optional[str] = None

    # Execution
    use_torch_compile: bool = True
    compile_mode: str = "default"
    compile_dynamic: bool = True
    resume_from: Optional[str] = None

    # Real-batch autotune helper
    auto_tune_tokens_per_batch: bool = False
    auto_tune_target_mem_frac: float = 0.80
    auto_tune_probe_steps: int = 16
    auto_tune_tokens_per_batch_max: int = 65536
    auto_tune_round_to: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        allowed = {fd.name for fd in fields(TrainConfig)}
        d_f = {k: v for k, v in d.items() if k in allowed}
        return TrainConfig(**d_f)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
