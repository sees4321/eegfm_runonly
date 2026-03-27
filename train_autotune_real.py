
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


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

EEGModelConfig = cfg_mod.EEGModelConfig
TrainConfig = cfg_mod.TrainConfig
run_train = train_mod.run_train

import torch
import argparse as _argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-batch tokens_per_batch autotune helper")
    p.add_argument("--model_cfg", type=str, required=True)
    p.add_argument("--train_cfg", type=str, required=True)
    p.add_argument("--shards_txt", type=str, default=None)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--min", dest="min_tokens", type=int, default=None)
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--round_to", type=int, default=None)
    p.add_argument("--target_mem_frac", type=float, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
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

    if not torch.cuda.is_available():
        raise RuntimeError("autotune requires CUDA")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    round_to = int(train_cfg.auto_tune_round_to)
    high = int(train_cfg.auto_tune_tokens_per_batch_max)
    if args.min_tokens is not None:
        low = int(args.min_tokens)
    else:
        low = int(round_to)
    low = max(int(round_to), (low // round_to) * round_to)
    high = max(low, (high // round_to) * round_to)
    best = 0

    while low <= high:
        mid = max(round_to, ((low + high) // 2 // round_to) * round_to)
        if mid <= 0:
            break

        probe_cfg_path = os.path.join("/tmp", f"eegfm_autotune_cfg_rank{rank}.json")
        probe_out_dir = os.path.join("/tmp", f"eegfm_autotune_out_rank{rank}")
        probe_cfg = TrainConfig.from_json(args.train_cfg)
        probe_cfg.shards_txt = train_cfg.shards_txt
        probe_cfg.tokens_per_batch = mid
        probe_cfg.max_steps = max(4, int(train_cfg.auto_tune_probe_steps))
        probe_cfg.save_every = 0
        probe_cfg.save_every_pass = False
        probe_cfg.use_wandb = False
        probe_cfg.output_dir = probe_out_dir
        probe_cfg.resume_from = None
        probe_cfg.save_json(probe_cfg_path)

        ok = True
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            run_train(_argparse.Namespace(
                model_cfg=args.model_cfg,
                train_cfg=probe_cfg_path,
                shards_txt=probe_cfg.shards_txt,
                output_dir=probe_out_dir,
                resume_from=None,
                tokens_per_batch=mid,
                tokens_per_update=probe_cfg.tokens_per_update,
                max_steps=probe_cfg.max_steps,
                lr=probe_cfg.lr,
                weight_decay=probe_cfg.weight_decay,
                num_workers=probe_cfg.num_workers,
                run_name=None,
                no_wandb=True,
            ))
            peak = torch.cuda.max_memory_reserved(device)
            total = torch.cuda.get_device_properties(device).total_memory
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

    if best <= 0:
        raise RuntimeError("autotune could not find a safe tokens_per_batch; lower --min / model size or check memory headroom")

    if rank == 0:
        print(json.dumps({
            "recommended_tokens_per_batch_per_rank": int(best),
            "search_min": int((args.min_tokens if args.min_tokens is not None else round_to) // round_to * round_to),
            "search_max": int(train_cfg.auto_tune_tokens_per_batch_max // round_to * round_to),
            "target_mem_frac": float(train_cfg.auto_tune_target_mem_frac),
        }, indent=2))


if __name__ == "__main__":
    main()
