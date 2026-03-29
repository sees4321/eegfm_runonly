
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from .augment import apply_student_augmentations
from .config import EEGModelConfig, TrainConfig
from .data import ShapeBatcher, build_webdataset, read_shards_txt
from .model import EEGEncoder, CrossAttentionPredictor
try:
    import webdataset as wds
except Exception:
    wds = None

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
except Exception:
    Accelerator = None


class ZProjector(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EMAUpdater:
    def __init__(self, teacher: nn.Module, student: nn.Module, m0: float):
        self.teacher_params = list(teacher.parameters())
        self.student_params = list(student.parameters())
        self.m = float(m0)

    def set_momentum(self, step: int, total_steps: int, m0: float, m1: float) -> None:
        progress = step / max(1, total_steps)
        self.m = float(m1 - (m1 - m0) * (0.5 * (1.0 + math.cos(math.pi * progress))))

    @torch.no_grad()
    def update(self) -> None:
        torch._foreach_mul_(self.teacher_params, self.m)
        torch._foreach_add_(self.teacher_params, self.student_params, alpha=(1.0 - self.m))


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=add_help)
    p.add_argument("--model_cfg", type=str, required=True)
    p.add_argument("--train_cfg", type=str, required=True)
    p.add_argument("--shards_txt", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--tokens_per_batch", type=int, default=None)
    p.add_argument("--tokens_per_update", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _distributed_sum(value: int, device: torch.device) -> int:
    t = torch.tensor([int(value)], device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def _distributed_mean(value: float, device: torch.device) -> float:
    t = torch.tensor([float(value)], device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


@torch.no_grad()
def compute_logspec_view(
    patches: torch.Tensor,
    fs: int,
    f_min: float = 1.0,
    f_max: float = 45.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    x = patches - patches.mean(dim=-1, keepdim=True)
    win = torch.hann_window(x.shape[-1], periodic=True, device=x.device, dtype=x.dtype)
    x = x * win[None, :]
    X = torch.fft.rfft(x, dim=-1)
    P = (X.real ** 2 + X.imag ** 2).clamp_min(eps)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / float(fs)).to(device=x.device, dtype=torch.float32)
    sel = (freqs >= float(f_min)) & (freqs <= float(f_max))
    logp = torch.log(P[:, sel])
    logp = logp - logp.mean(dim=-1, keepdim=True)
    logp = logp / (logp.std(dim=-1, keepdim=True).clamp_min(1e-6))
    return logp


def relational_kl_loss(z: torch.Tensor, s: torch.Tensor, tau_z: float = 0.1, tau_s: float = 0.1) -> torch.Tensor:
    z_n = F.normalize(z, dim=-1)
    s_n = F.normalize(s, dim=-1)
    logits_z = (z_n @ z_n.T) / tau_z
    logits_s = (s_n @ s_n.T) / tau_s
    eye = torch.eye(logits_z.shape[0], device=z.device, dtype=torch.bool)
    logits_z = logits_z.masked_fill(eye, float("-inf"))
    logits_s = logits_s.masked_fill(eye, float("-inf"))
    log_p = F.log_softmax(logits_z, dim=-1).masked_fill(eye, 0.0)
    log_q = F.log_softmax(logits_s, dim=-1).detach().masked_fill(eye, 0.0)
    return F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)


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
    k_idx = max(1, int(round(quantile * T)))
    robust_amp = x32.abs().kthvalue(k_idx, dim=-1, keepdim=True).values
    gain = (target_amp / robust_amp.clamp_min(amp_floor)).clamp(max=gain_max)
    x32 = x32 * gain
    if clip and clip > 0:
        x32 = x32.clamp(-clip, clip)
    return x32.to(dtype=x.dtype)


def gather_channel_embeddings(x: torch.Tensor, c_idx: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    B, C, D = x.shape
    B2, L = c_idx.shape
    assert B == B2
    idx = c_idx[..., None].expand(B, L, D)
    out = x.gather(dim=1, index=idx)
    return out.masked_fill(pad[..., None], 0.0)


def token_wcc_lr(tokens_next: int, warmup_tokens: int, total_tokens: int, cooldown_tokens: int, base_lr: float, min_lr: float = 0.0) -> float:
    t = max(0, int(tokens_next))
    T = max(1, int(total_tokens))
    W = min(max(0, int(warmup_tokens)), T)
    C = min(max(0, int(cooldown_tokens)), max(0, T - W))
    stable = max(0, T - W - C)

    if W > 0 and t < W:
        return float(base_lr) * float(t) / float(max(1, W))
    if t < (W + stable):
        return float(base_lr)
    if C > 0 and t < T:
        td = t - (W + stable)
        frac = float(td) / float(max(1, C))
        return float(base_lr) + (float(min_lr) - float(base_lr)) * frac
    return float(min_lr)


def maybe_compile_modules(student: EEGEncoder, predictor: CrossAttentionPredictor, train_cfg: TrainConfig) -> None:
    if not bool(train_cfg.use_torch_compile):
        return
    mode = str(train_cfg.compile_mode)
    dynamic = bool(train_cfg.compile_dynamic)
    for blk in student.blocks:
        if hasattr(blk, "attn"):
            blk.attn = torch.compile(blk.attn, mode=mode, dynamic=dynamic)
        if hasattr(blk, "attn_t"):
            blk.attn_t = torch.compile(blk.attn_t, mode=mode, dynamic=dynamic)
        if hasattr(blk, "attn_s"):
            blk.attn_s = torch.compile(blk.attn_s, mode=mode, dynamic=dynamic)
        if hasattr(blk, "mlp"):
            blk.mlp = torch.compile(blk.mlp, mode=mode, dynamic=dynamic)
    for blk in predictor.blocks:
        blk.xattn = torch.compile(blk.xattn, mode=mode, dynamic=dynamic)
        blk.mlp = torch.compile(blk.mlp, mode=mode, dynamic=dynamic)


def save_checkpoint(
    ckpt_dir: str,
    student: EEGEncoder,
    teacher: EEGEncoder,
    predictor: CrossAttentionPredictor,
    rel_proj: Optional[ZProjector],
    optimizer: AdamW,
    trainer_state: Dict[str, int],
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    student.save_pretrained(os.path.join(ckpt_dir, "student"))
    teacher.save_pretrained(os.path.join(ckpt_dir, "teacher"))
    torch.save(predictor.state_dict(), os.path.join(ckpt_dir, "predictor.pt"))
    if rel_proj is not None:
        torch.save(rel_proj.state_dict(), os.path.join(ckpt_dir, "rel_projector.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)


def load_checkpoint(
    ckpt_dir: str,
    student: EEGEncoder,
    teacher: EEGEncoder,
    predictor: CrossAttentionPredictor,
    rel_proj: Optional[ZProjector],
    optimizer: AdamW,
) -> Dict[str, int]:
    student_sd = torch.load(os.path.join(ckpt_dir, "student", "pytorch_model.bin"), map_location="cpu", weights_only=True)
    teacher_sd = torch.load(os.path.join(ckpt_dir, "teacher", "pytorch_model.bin"), map_location="cpu", weights_only=True)
    pred_sd = torch.load(os.path.join(ckpt_dir, "predictor.pt"), map_location="cpu", weights_only=True)
    opt_sd = torch.load(os.path.join(ckpt_dir, "optimizer.pt"), map_location="cpu", weights_only=False)

    student.load_state_dict(student_sd, strict=True)
    teacher.load_state_dict(teacher_sd, strict=True)
    predictor.load_state_dict(pred_sd, strict=True)
    if rel_proj is not None and os.path.exists(os.path.join(ckpt_dir, "rel_projector.pt")):
        rel_sd = torch.load(os.path.join(ckpt_dir, "rel_projector.pt"), map_location="cpu", weights_only=True)
        rel_proj.load_state_dict(rel_sd, strict=True)
    optimizer.load_state_dict(opt_sd)
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def count_params(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


def run_train(args: argparse.Namespace) -> Dict[str, str]:
    if Accelerator is None:
        raise RuntimeError("accelerate is not installed. pip install accelerate")
    if wds is None:
        raise RuntimeError("webdataset is not installed. pip install webdataset")

    model_cfg = EEGModelConfig.from_json(args.model_cfg)
    train_cfg = TrainConfig.from_json(args.train_cfg)

    if args.shards_txt is not None:
        train_cfg.shards_txt = args.shards_txt
    if args.output_dir is not None:
        train_cfg.output_dir = args.output_dir
    if args.resume_from is not None:
        train_cfg.resume_from = args.resume_from
    if args.tokens_per_batch is not None:
        train_cfg.tokens_per_batch = int(args.tokens_per_batch)
    if args.tokens_per_update is not None:
        train_cfg.tokens_per_update = int(args.tokens_per_update)
    if args.max_steps is not None:
        train_cfg.max_steps = int(args.max_steps)
    if args.lr is not None:
        train_cfg.lr = float(args.lr)
    if args.weight_decay is not None:
        train_cfg.weight_decay = float(args.weight_decay)
    if args.num_workers is not None:
        train_cfg.num_workers = int(args.num_workers)
    if args.run_name is not None:
        train_cfg.run_name = args.run_name
    if args.no_wandb:
        train_cfg.use_wandb = False

    if not train_cfg.shards_txt:
        raise ValueError("train_cfg.shards_txt (or --shards_txt) is required in the run-only project")

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=train_cfg.mixed_precision,
        log_with="wandb" if train_cfg.use_wandb else None,
        project_dir=train_cfg.output_dir,
    )
    dev = accelerator.device

    seed = int(train_cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed, device_specific=True)

    if bool(train_cfg.torch_deterministic):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = bool(train_cfg.cudnn_benchmark)

    if accelerator.is_main_process:
        os.makedirs(train_cfg.output_dir, exist_ok=True)
        model_cfg.save_json(os.path.join(train_cfg.output_dir, "model_config.json"))
        train_cfg.save_json(os.path.join(train_cfg.output_dir, "train_config.json"))

    if train_cfg.use_wandb:
        accelerator.init_trackers(
            train_cfg.wandb_project,
            config={**model_cfg.to_dict(), **train_cfg.to_dict()},
            init_kwargs={"wandb": {"name": train_cfg.run_name}} if train_cfg.run_name else None,
        )

    student = EEGEncoder(model_cfg).to(dev)
    predictor = CrossAttentionPredictor(model_cfg).to(dev)
    rel_proj = None
    use_rel = str(train_cfg.spec_aux_mode).lower() == "relational" and float(train_cfg.spec_rel_weight) > 0
    if use_rel:
        rel_proj = ZProjector(model_cfg.d_model, int(train_cfg.spec_rel_proj_dim)).to(dev)

    teacher = copy.deepcopy(student).to(dev)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    student_raw = student
    predictor_raw = predictor
    maybe_compile_modules(student_raw, predictor_raw, train_cfg)

    params = list(student.parameters()) + list(predictor.parameters())
    if rel_proj is not None:
        params += list(rel_proj.parameters())
    optimizer = AdamW(params, lr=train_cfg.lr, betas=train_cfg.betas, weight_decay=train_cfg.weight_decay)

    student, predictor, optimizer = accelerator.prepare(student, predictor, optimizer)
    if rel_proj is not None:
        rel_proj = accelerator.prepare(rel_proj)

    teacher.to(dev)
    ema_updater = EMAUpdater(teacher, accelerator.unwrap_model(student), train_cfg.ema_momentum)

    trainer_state = {
        "global_step": 0,
        "passes_completed": 0,
        "tokens_seen_total": 0,
    }
    if train_cfg.resume_from:
        ckpt_dir = str(Path(train_cfg.resume_from).expanduser().resolve())
        trainer_state = load_checkpoint(
            ckpt_dir,
            accelerator.unwrap_model(student),
            teacher,
            accelerator.unwrap_model(predictor),
            accelerator.unwrap_model(rel_proj) if rel_proj is not None else None,
            optimizer,
        )
        accelerator.wait_for_everyone()

    patch_samples = int(round(model_cfg.sample_rate * model_cfg.patch_seconds))
    hop_samples = max(1, int(round(model_cfg.sample_rate * model_cfg.patch_hop_seconds)))
    shards = read_shards_txt(train_cfg.shards_txt)

    ds = build_webdataset(
        shards=shards,
        cache_dir=train_cfg.cache_dir,
        shard_shuffle=train_cfg.shard_shuffle,
        sample_shuffle=train_cfg.sample_shuffle,
        max_tokens=model_cfg.max_tokens,
        patch_samples=patch_samples,
        hop_samples=hop_samples,
        cache_max_bytes=train_cfg.cache_max_bytes,
        post_split_shuffle=train_cfg.post_split_shuffle,
        eviction_interval=train_cfg.eviction_interval,
        seed=train_cfg.seed,
    )
    ds_batched = ShapeBatcher(
        dataset=ds,
        tokens_per_batch=train_cfg.tokens_per_batch,
        max_samples_per_batch=train_cfg.max_samples_per_batch,
        patch_samples=patch_samples,
        hop_samples=hop_samples,
        target_mask_cfg=dict(
            mask_time_prob=float(train_cfg.mask_time_prob),
            mask_spatial_prob=float(train_cfg.mask_spatial_prob),
            time_ratio_range=(float(train_cfg.time_mask_ratio_min), float(train_cfg.time_mask_ratio_max)),
            spatial_ratio_range=(float(train_cfg.spatial_mask_ratio_min), float(train_cfg.spatial_mask_ratio_max)),
            mask_dilate_time=int(train_cfg.mask_dilate_time),
        ),
        max_wait_samples=5000,
        flush_check_every=256,
        max_pending_samples=512,
        max_pending_tokens=0,
        shuffle_within_bucket=True,
        yield_incomplete=False,
    )
    loader = wds.WebLoader(
        ds_batched,
        batch_size=None,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        persistent_workers=(train_cfg.num_workers > 0),
        prefetch_factor=4 if train_cfg.num_workers > 0 else None,
    )

    if accelerator.is_main_process:
        enc_params = count_params(accelerator.unwrap_model(student))
        pred_params = count_params(accelerator.unwrap_model(predictor))
        rel_params = count_params(accelerator.unwrap_model(rel_proj)) if rel_proj is not None else 0
        print(f"[params] encoder={enc_params/1e6:.3f}M predictor={pred_params/1e6:.3f}M aux={rel_params/1e6:.3f}M")

    budget_tokens = int(train_cfg.tokens_per_update)
    total_sched_tokens = int(train_cfg.max_steps) * budget_tokens
    warmup_tokens = int(train_cfg.warmup_steps) * budget_tokens
    cooldown_tokens = int(float(train_cfg.lr_cooldown_frac) * float(total_sched_tokens))
    world_size = accelerator.num_processes

    global_step = int(trainer_state.get("global_step", 0))
    passes_completed = int(trainer_state.get("passes_completed", 0))
    tokens_seen_total = int(trainer_state.get("tokens_seen_total", 0))

    opt_zero = True
    if opt_zero:
        optimizer.zero_grad(set_to_none=True)

    accum_tokens_global = 0
    accum_tokens_eff_global = 0
    pass_input_tokens_global = 0
    pass_target_tokens_global = 0
    pass_context_tokens_global = 0

    it = iter(loader)
    pbar = tqdm(total=int(train_cfg.max_steps), disable=not accelerator.is_local_main_process)
    if global_step > 0:
        pbar.update(global_step)

    start_time = time.time()
    last_logged_step = global_step

    while global_step < int(train_cfg.max_steps):
        try:
            batch = next(it)
            local_exhausted = 0
        except StopIteration:
            batch = None
            local_exhausted = 1

        exhausted_any = _distributed_sum(local_exhausted, dev) > 0
        if exhausted_any:
            passes_completed += 1
            if accelerator.is_main_process:
                print(
                    f"[data] pass {passes_completed} completed. Step: {global_step}\n"
                    f"Input Tokens: {pass_input_tokens_global}, Target Tokens: {pass_target_tokens_global}, "
                    f"Context Tokens: {pass_context_tokens_global}"
                )
            if bool(train_cfg.save_every_pass) and global_step > 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ckpt_dir = os.path.join(train_cfg.output_dir, f"pass_{passes_completed:03d}_step_{global_step:07d}")
                    save_checkpoint(
                        ckpt_dir,
                        accelerator.unwrap_model(student),
                        teacher,
                        accelerator.unwrap_model(predictor),
                        accelerator.unwrap_model(rel_proj) if rel_proj is not None else None,
                        optimizer,
                        {
                            "global_step": global_step,
                            "passes_completed": passes_completed,
                            "tokens_seen_total": tokens_seen_total,
                        },
                    )
            pass_input_tokens_global = 0
            pass_target_tokens_global = 0
            pass_context_tokens_global = 0
            it = iter(loader)
            continue

        x = batch["eeg"].to(dev, non_blocking=True)
        coords = batch["coord"].to(dev, non_blocking=True)
        target_mask = batch["target_mask"].to(dev, non_blocking=True)
        context_mask = batch["context_mask"].to(dev, non_blocking=True)
        c_ctx = batch["c_ctx"].to(dev, non_blocking=True)
        t_ctx = batch["t_ctx"].to(dev, non_blocking=True)
        pad_ctx = batch["pad_ctx"].to(dev, non_blocking=True)
        c_tgt = batch["c_tgt"].to(dev, non_blocking=True)
        t_tgt = batch["t_tgt"].to(dev, non_blocking=True)
        pad_tgt = batch["pad_tgt"].to(dev, non_blocking=True)

        B = x.shape[0]
        P_max = int(batch["P_max_cpu"])
        x = rescale_small_segments(x, target_amp=1.0, quantile=0.90, amp_floor=1e-4, gain_max=200.0, clip=15.0)

        local_input_tokens = int(batch["valid_tokens"])
        local_target_tokens = int(batch["target_tokens"])
        local_context_tokens = int(batch["context_tokens"])

        global_input_tokens = _distributed_sum(local_input_tokens, dev)
        global_target_tokens = _distributed_sum(local_target_tokens, dev)
        global_context_tokens = _distributed_sum(local_context_tokens, dev)

        pass_input_tokens_global += global_input_tokens
        pass_target_tokens_global += global_target_tokens
        pass_context_tokens_global += global_context_tokens

        if accum_tokens_global == 0 and global_target_tokens <= 0:
            will_step = True
        else:
            will_step = (accum_tokens_global + global_target_tokens) >= max(1, budget_tokens)

        no_sync = (not will_step) and (world_size > 1)
        weight = 1.0
        tokens_eff_this_global = global_target_tokens
        if will_step and budget_tokens > 0:
            remain = max(0, budget_tokens - accum_tokens_global)
            if remain > 0 and global_target_tokens > remain:
                weight = float(remain) / float(global_target_tokens)
                tokens_eff_this_global = int(remain)

        with ExitStack() as stack:
            if no_sync:
                if hasattr(student, "no_sync"):
                    stack.enter_context(student.no_sync())
                if hasattr(predictor, "no_sync"):
                    stack.enter_context(predictor.no_sync())
                if rel_proj is not None and hasattr(rel_proj, "no_sync"):
                    stack.enter_context(rel_proj.no_sync())

            x_aug = apply_student_augmentations(
                x,
                gain_min=train_cfg.aug_gain_min,
                gain_max=train_cfg.aug_gain_max,
                channel_gain_std=train_cfg.aug_channel_gain_std,
                noise_std_min=train_cfg.aug_noise_std_min,
                noise_std_max=train_cfg.aug_noise_std_max,
                channel_drop_prob=train_cfg.aug_channel_drop_prob,
            )

            with accelerator.autocast():
                student_raw = accelerator.unwrap_model(student)
                predictor_raw = accelerator.unwrap_model(predictor)

                coord_ch_student = student_raw.coord_embed(coords)
                tok_ctx, pad_ctx2, rope_ctx, chan_ctx = student_raw.embed_from_indices(
                    x=x_aug,
                    coords=coords,
                    c_idx=c_ctx,
                    t_idx=t_ctx,
                    pad=pad_ctx,
                    coord_ch=coord_ch_student,
                )
                z_ctx = student(tok_ctx, padding_mask=pad_ctx2, rope_pos=rope_ctx, chan_idx=chan_ctx, coords=coords, grid_patches=P_max)

                with torch.no_grad():
                    coord_ch_teacher = teacher.coord_embed(coords)
                    patches_view_t = teacher.extract_patches_view(x)
                    c_safe_t = c_tgt.clamp(min=0)
                    t_safe_t = t_tgt.clamp(min=0)
                    b_idx_t = torch.arange(B, device=x.device)[:, None].expand(B, c_safe_t.shape[1])
                    cached_tgt_patches = patches_view_t[b_idx_t, c_safe_t, t_safe_t]

                    tok_tgt, pad_tgt2, rope_tgt, chan_tgt = teacher.embed_from_indices(
                        x=x,
                        coords=coords,
                        c_idx=c_tgt,
                        t_idx=t_tgt,
                        pad=pad_tgt,
                        coord_ch=coord_ch_teacher,
                    )
                    z_tgt = teacher(tok_tgt, padding_mask=pad_tgt2, rope_pos=rope_tgt, chan_idx=chan_tgt, coords=coords)

                coord_tgt = gather_channel_embeddings(coord_ch_student, c_tgt.clamp(min=0), pad_tgt)
                pred_tgt = predictor(
                    ctx=z_ctx,
                    ctx_pad=pad_ctx2,
                    rope_ctx=rope_ctx,
                    tgt_coord_emb=coord_tgt,
                    tgt_pad=pad_tgt,
                    rope_tgt=rope_tgt,
                )

                valid_tgt = ~pad_tgt
                num_tgt_local = valid_tgt.sum().float().clamp_min(1.0)

                diff = torch.abs(pred_tgt - z_tgt).masked_fill(pad_tgt[..., None], 0.0)
                l1_sum_local = diff.sum().float()
                loss_sum_local = l1_sum_local

                spec_rel_loss = None
                spec_lam = 0.0
                if rel_proj is not None:
                    patches_flat = cached_tgt_patches[valid_tgt].to(torch.float32)
                    spec_target = compute_logspec_view(
                        patches_flat,
                        fs=model_cfg.sample_rate,
                        f_min=1.0,
                        f_max=45.0,
                    )
                    z_flat = pred_tgt[valid_tgt]
                    M = int(train_cfg.spec_rel_subsample_tokens)
                    if z_flat.shape[0] > M:
                        idx = torch.randperm(z_flat.shape[0], device=x.device)[:M]
                        z_flat = z_flat[idx]
                        spec_target = spec_target[idx]
                    z_rel = rel_proj(z_flat)
                    spec_rel_loss = relational_kl_loss(
                        z=z_rel,
                        s=spec_target,
                        tau_z=float(train_cfg.spec_rel_tau_z),
                        tau_s=float(train_cfg.spec_rel_tau_s),
                    )
                    warm = int(train_cfg.spec_rel_warmup_steps)
                    ramp = int(train_cfg.spec_rel_ramp_steps)
                    if global_step < warm:
                        spec_lam = 0.0
                    elif ramp > 0:
                        u = min(1.0, float(global_step - warm) / float(ramp))
                        spec_lam = float(train_cfg.spec_rel_weight) * u
                    else:
                        spec_lam = float(train_cfg.spec_rel_weight)
                    loss_sum_local = loss_sum_local + spec_lam * spec_rel_loss

                loss_scaled = (loss_sum_local * float(weight) * float(world_size)) / (float(max(1, budget_tokens)) * float(model_cfg.d_model))

            accelerator.backward(loss_scaled)

        accum_tokens_global += int(global_target_tokens)
        accum_tokens_eff_global += int(tokens_eff_this_global)

        if will_step:
            tokens_next = int(tokens_seen_total) + int(accum_tokens_eff_global)
            lr_now = token_wcc_lr(
                tokens_next=tokens_next,
                warmup_tokens=warmup_tokens,
                total_tokens=total_sched_tokens,
                cooldown_tokens=cooldown_tokens,
                base_lr=float(train_cfg.lr),
                min_lr=float(train_cfg.min_lr),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            if float(train_cfg.grad_clip) > 0:
                accelerator.clip_grad_norm_(params, float(train_cfg.grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            ema_updater.set_momentum(global_step, int(train_cfg.max_steps), float(train_cfg.ema_momentum), float(train_cfg.ema_momentum_final))
            ema_updater.update()

            tokens_seen_total = tokens_next
            global_step += 1
            pbar.update(1)

            l1_den_local = max(1.0, float(local_target_tokens) * float(model_cfg.d_model))
            l1_mean_local = float(l1_sum_local.detach().item()) / l1_den_local
            l1_mean = _distributed_mean(l1_mean_local, dev)

            if (global_step - last_logged_step) >= int(train_cfg.log_every):
                step_time = (time.time() - start_time) / max(1, global_step - last_logged_step)
                logs = {
                    "train/l1_mean": l1_mean,
                    "train/lr": float(lr_now),
                    "train/ema_momentum": float(ema_updater.m),
                    "train/tokens_seen_total": int(tokens_seen_total),
                    "train/pass_input_tokens": int(pass_input_tokens_global),
                    "train/pass_target_tokens": int(pass_target_tokens_global),
                    "train/pass_context_tokens": int(pass_context_tokens_global),
                    "train/tokens_per_update": int(train_cfg.tokens_per_update),
                    "train/tokens_per_batch_per_rank": int(train_cfg.tokens_per_batch),
                    "system/step_time_sec": float(step_time),
                    "system/world_size": int(world_size),
                }
                if spec_rel_loss is not None:
                    logs["train/spec_rel_loss"] = _distributed_mean(float(spec_rel_loss.detach().item()), dev)
                    logs["train/spec_rel_weight"] = float(spec_lam)

                if accelerator.is_main_process and not train_cfg.use_wandb:
                    print(
                        f"[step {global_step:07d}] l1={logs['train/l1_mean']:.6f} "
                        f"lr={logs['train/lr']:.3e} ema={logs['train/ema_momentum']:.6f} "
                        f"step_time={logs['system/step_time_sec']:.3f}s"
                    )
                accelerator.log(logs, step=global_step)
                start_time = time.time()
                last_logged_step = global_step

            if (int(train_cfg.save_every) > 0) and (global_step % int(train_cfg.save_every) == 0):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ckpt_dir = os.path.join(train_cfg.output_dir, f"step_{global_step:07d}")
                    save_checkpoint(
                        ckpt_dir,
                        accelerator.unwrap_model(student),
                        teacher,
                        accelerator.unwrap_model(predictor),
                        accelerator.unwrap_model(rel_proj) if rel_proj is not None else None,
                        optimizer,
                        {
                            "global_step": global_step,
                            "passes_completed": passes_completed,
                            "tokens_seen_total": tokens_seen_total,
                        },
                    )

            accum_tokens_global = 0
            accum_tokens_eff_global = 0

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(train_cfg.output_dir, "final")
        save_checkpoint(
            final_dir,
            accelerator.unwrap_model(student),
            teacher,
            accelerator.unwrap_model(predictor),
            accelerator.unwrap_model(rel_proj) if rel_proj is not None else None,
            optimizer,
            {
                "global_step": global_step,
                "passes_completed": passes_completed,
                "tokens_seen_total": tokens_seen_total,
            },
        )
    accelerator.wait_for_everyone()
    return {
        "output_dir": str(train_cfg.output_dir),
        "final_dir": os.path.join(train_cfg.output_dir, "final"),
        "student_dir": os.path.join(train_cfg.output_dir, "final", "student"),
        "teacher_dir": os.path.join(train_cfg.output_dir, "final", "teacher"),
        "predictor_path": os.path.join(train_cfg.output_dir, "final", "predictor.pt"),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_train(args)


if __name__ == "__main__":
    main()
