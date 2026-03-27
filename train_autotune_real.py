
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

from .config import EEGModelConfig, TrainConfig
from .train import parse_args as _unused  # noqa: F401  # keep package relative import healthy


def _bootstrap():
    if __package__:
        return __package__
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent.parent))
    return here.parent.name


PKG = _bootstrap()
from importlib import import_module
cfg_mod = import_module(f"{PKG}.config")
train_mod = import_module(f"{PKG}.train")
model_mod = import_module(f"{PKG}.model")
data_mod = import_module(f"{PKG}.data")

EEGModelConfig = cfg_mod.EEGModelConfig
TrainConfig = cfg_mod.TrainConfig
run_train = train_mod.run_train
Accelerator = import_module("accelerate").Accelerator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-batch tokens_per_batch autotune helper")
    p.add_argument("--model_cfg", type=str, required=True)
    p.add_argument("--train_cfg", type=str, required=True)
    p.add_argument("--shards_txt", type=str, default=None)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--round_to", type=int, default=None)
    p.add_argument("--target_mem_frac", type=float, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    model_cfg = EEGModelConfig.from_json(args.model_cfg)
    train_cfg = TrainConfig.from_json(args.train_cfg)
    if args.shards_txt is not None:
        train_cfg.shards_txt = args.shards_txt
    if args.start is not None:
        train_cfg.tokens_per_batch = int(args.start)
    if args.max is not None:
        train_cfg.auto_tune_tokens_per_batch_max = int(args.max)
    if args.round_to is not None:
        train_cfg.auto_tune_round_to = int(args.round_to)
    if args.target_mem_frac is not None:
        train_cfg.auto_tune_target_mem_frac = float(args.target_mem_frac)
    train_cfg.use_wandb = False

    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision=train_cfg.mixed_precision, log_with=None, project_dir=None)
    if accelerator.device.type != "cuda":
        raise RuntimeError("autotune requires CUDA")

    # Conservative monotonic search using a tiny real run.
    low = int(train_cfg.tokens_per_batch)
    high = int(train_cfg.auto_tune_tokens_per_batch_max)
    round_to = int(train_cfg.auto_tune_round_to)
    best = low

    while low <= high:
        mid = max(round_to, ((low + high) // 2 // round_to) * round_to)
        if mid <= 0:
            break

        probe_cfg_path = os.path.join("/tmp", f"eegfm_autotune_rank{accelerator.process_index}.json")
        train_cfg.tokens_per_batch = mid
        train_cfg.max_steps = max(4, int(train_cfg.auto_tune_probe_steps))
        train_cfg.save_every = 0
        train_cfg.save_every_pass = False
        train_cfg.output_dir = os.path.join("/tmp", f"eegfm_autotune_rank{accelerator.process_index}")
        train_cfg.resume_from = None
        train_cfg.save_json(probe_cfg_path)

        ok = True
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(accelerator.device)
            run_train(argparse.Namespace(
                model_cfg=args.model_cfg,
                train_cfg=probe_cfg_path,
                shards_txt=train_cfg.shards_txt,
                output_dir=train_cfg.output_dir,
                resume_from=None,
                tokens_per_batch=mid,
                tokens_per_update=train_cfg.tokens_per_update,
                max_steps=train_cfg.max_steps,
                lr=train_cfg.lr,
                weight_decay=train_cfg.weight_decay,
                num_workers=train_cfg.num_workers,
                run_name=None,
                no_wandb=True,
            ))
            peak = torch.cuda.max_memory_reserved(accelerator.device)
            total = torch.cuda.get_device_properties(accelerator.device).total_memory
            ok = peak <= int(total * float(train_cfg.auto_tune_target_mem_frac))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                ok = False
                torch.cuda.empty_cache()
            else:
                raise

        if ok:
            best = mid
            low = mid + round_to
        else:
            high = mid - round_to

    if accelerator.is_main_process:
        print(json.dumps({
            "recommended_tokens_per_batch_per_rank": int(best),
            "target_mem_frac": float(train_cfg.auto_tune_target_mem_frac),
        }, indent=2))


if __name__ == "__main__":
    main()
