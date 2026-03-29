from __future__ import annotations

import argparse
import json
import os
import subprocess
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

TrainConfig = cfg_mod.TrainConfig
run_train = train_mod.run_train

import argparse as _argparse
import torch
import torch.distributed as dist


RESULT_PREFIX = "AUTOTUNE_PROBE_RESULT "


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
    p.add_argument("--num_processes", type=int, default=2)
    p.add_argument("--probe_mode", action="store_true")
    p.add_argument("--probe_tokens_per_batch", type=int, default=None)
    return p


def _round_down(x: int, q: int) -> int:
    return max(q, (int(x) // int(q)) * int(q))


def _probe_once(args: argparse.Namespace) -> None:
    if args.probe_tokens_per_batch is None:
        raise ValueError("--probe_tokens_per_batch is required in --probe_mode")

    train_cfg = TrainConfig.from_json(args.train_cfg)
    if args.shards_txt is not None:
        train_cfg.shards_txt = args.shards_txt
    train_cfg.tokens_per_batch = int(args.probe_tokens_per_batch)
    train_cfg.max_steps = max(4, int(train_cfg.auto_tune_probe_steps))
    train_cfg.save_every = 0
    train_cfg.save_every_pass = False
    train_cfg.use_wandb = False
    train_cfg.resume_from = None

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    probe_cfg_path = os.path.join("/tmp", f"eegfm_autotune_cfg_rank{os.environ.get('RANK', '0')}.json")
    probe_out_dir = os.path.join("/tmp", f"eegfm_autotune_out_rank{os.environ.get('RANK', '0')}")
    train_cfg.output_dir = probe_out_dir
    train_cfg.save_json(probe_cfg_path)

    ok = True
    err_msg = None
    try:
        run_train(_argparse.Namespace(
            model_cfg=args.model_cfg,
            train_cfg=probe_cfg_path,
            shards_txt=train_cfg.shards_txt,
            output_dir=probe_out_dir,
            resume_from=None,
            tokens_per_batch=train_cfg.tokens_per_batch,
            tokens_per_update=train_cfg.tokens_per_update,
            max_steps=train_cfg.max_steps,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            num_workers=train_cfg.num_workers,
            run_name=None,
            no_wandb=True,
        ))
    except RuntimeError as e:
        msg = str(e)
        err_msg = msg
        if "out of memory" in msg.lower() or "cuda out of memory" in msg.lower():
            ok = False
        else:
            raise
    finally:
        peak = int(torch.cuda.max_memory_reserved(device))
        total = int(torch.cuda.get_device_properties(device).total_memory)
        peak_t = torch.tensor([peak], device=device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
        peak = int(peak_t.item())
        if ok:
            ok = peak <= int(total * float(train_cfg.auto_tune_target_mem_frac))
        result = {
            "probe_tokens_per_batch": int(train_cfg.tokens_per_batch),
            "peak_reserved": int(peak),
            "total_memory": int(total),
            "target_mem_frac": float(train_cfg.auto_tune_target_mem_frac),
            "ok": bool(ok),
            "error": err_msg,
        }
        if int(os.environ.get("RANK", "0")) == 0:
            print(RESULT_PREFIX + json.dumps(result), flush=True)
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    if not ok:
        raise SystemExit(2)


def _launch_probe(args: argparse.Namespace, mid: int) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(int(args.num_processes)),
        "-m",
        f"{PKG}.train_autotune_real",
        "--probe_mode",
        "--model_cfg",
        args.model_cfg,
        "--train_cfg",
        args.train_cfg,
        "--probe_tokens_per_batch",
        str(int(mid)),
    ]
    if args.shards_txt is not None:
        cmd.extend(["--shards_txt", args.shards_txt])

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout

    result = None
    for line in out.splitlines():
        if line.startswith(RESULT_PREFIX):
            payload = line[len(RESULT_PREFIX):].strip()
            try:
                result = json.loads(payload)
            except json.JSONDecodeError:
                pass

    if result is None:
        lower = out.lower()
        if "out of memory" in lower or "cuda out of memory" in lower:
            result = {
                "probe_tokens_per_batch": int(mid),
                "ok": False,
                "error": "OOM",
                "raw_output": out,
            }
        else:
            raise RuntimeError(
                "autotune probe failed without parsable result.\n"
                f"candidate={mid}\n"
                f"exitcode={proc.returncode}\n"
                f"---- probe output ----\n{out}"
            )
    result["returncode"] = int(proc.returncode)
    result["raw_output"] = out
    return result


def _controller(args: argparse.Namespace) -> None:
    train_cfg = TrainConfig.from_json(args.train_cfg)
    if args.start is not None:
        train_cfg.tokens_per_batch = int(args.start)
    if args.shards_txt is not None:
        train_cfg.shards_txt = args.shards_txt
    if args.max is not None:
        train_cfg.auto_tune_tokens_per_batch_max = int(args.max)
    if args.round_to is not None:
        train_cfg.auto_tune_round_to = int(args.round_to)
    if args.target_mem_frac is not None:
        train_cfg.auto_tune_target_mem_frac = float(args.target_mem_frac)

    q = int(train_cfg.auto_tune_round_to)
    high = _round_down(int(train_cfg.auto_tune_tokens_per_batch_max), q)
    low = int(args.min_tokens) if args.min_tokens is not None else q
    low = _round_down(low, q)
    high = max(low, high)
    best = 0
    history = []

    while low <= high:
        mid = _round_down((low + high) // 2, q)
        res = _launch_probe(args, mid)
        ok = bool(res.get("ok", False)) and int(res.get("returncode", 0)) == 0
        history.append({
            "probe_tokens_per_batch": int(mid),
            "ok": bool(ok),
            "peak_reserved": int(res.get("peak_reserved", -1)) if "peak_reserved" in res else None,
            "total_memory": int(res.get("total_memory", -1)) if "total_memory" in res else None,
            "returncode": int(res.get("returncode", -1)),
            "error": res.get("error"),
        })
        print(json.dumps(history[-1]), flush=True)
        if ok:
            best = mid
            low = mid + q
        else:
            high = mid - q

    if best <= 0:
        raise RuntimeError("autotune could not find a safe tokens_per_batch; lower --min / model size or check memory headroom")

    print(json.dumps({
        "recommended_tokens_per_batch_per_rank": int(best),
        "search_min": int(_round_down(int(args.min_tokens) if args.min_tokens is not None else q, q)),
        "search_max": int(_round_down(int(train_cfg.auto_tune_tokens_per_batch_max), q)),
        "target_mem_frac": float(train_cfg.auto_tune_target_mem_frac),
        "history": history,
    }, indent=2), flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.probe_mode:
        _probe_once(args)
    else:
        _controller(args)


if __name__ == "__main__":
    main()
