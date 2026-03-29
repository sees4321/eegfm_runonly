#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import os
import tarfile
from typing import Dict, List, Tuple

EXPECTED_EEG = {"eeg.npy"}
EXPECTED_COORD = {"coords.npy"}
KNOWN_EEG = {"eeg.npy", "eeg.npz", "eeg", "eeg.pyd", "x.npy", "x.npz", "x", "signal.npy", "signal"}
KNOWN_COORD = {"coords.npy", "coord.npy", "coords.npz", "coord.npz", "coords", "coord", "coords.pyd", "coord.pyd", "xyz.npy", "xyz"}


def read_shards_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def parse_member(name: str) -> Tuple[str, str]:
    base = os.path.basename(name)
    parts = base.split('.')
    if len(parts) < 2:
        return base, ""
    sample_id = parts[0]
    field = '.'.join(parts[1:])
    return sample_id, field


def scan_one_tar(path: str, max_examples_per_tar: int = 20) -> Dict[str, object]:
    samples = collections.defaultdict(set)
    with tarfile.open(path, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            sid, field = parse_member(m.name)
            if field:
                samples[sid].add(field)

    bad = []
    field_hist = collections.Counter()
    for sid, fields in samples.items():
        for f in fields:
            field_hist[f] += 1
        has_eeg = len(fields & KNOWN_EEG) > 0
        has_coord = len(fields & KNOWN_COORD) > 0
        if (not has_eeg) or (not has_coord) or (EXPECTED_EEG.isdisjoint(fields)) or (EXPECTED_COORD.isdisjoint(fields)):
            bad.append({
                "sample_id": sid,
                "fields": sorted(fields),
                "has_any_eeg_alias": has_eeg,
                "has_any_coord_alias": has_coord,
                "has_expected_eeg": not EXPECTED_EEG.isdisjoint(fields),
                "has_expected_coord": not EXPECTED_COORD.isdisjoint(fields),
            })
            if len(bad) >= max_examples_per_tar:
                break

    return {
        "tar": path,
        "num_samples": len(samples),
        "num_bad_examples": len(bad),
        "bad_examples": bad,
        "top_fields": field_hist.most_common(20),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_txt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_examples_per_tar", type=int, default=20)
    ap.add_argument("--stop_after_bad_tars", type=int, default=0)
    args = ap.parse_args()

    shards = read_shards_txt(args.shards_txt)
    bad_tars = 0
    total = []
    for i, shard in enumerate(shards, 1):
        rep = scan_one_tar(shard, max_examples_per_tar=args.max_examples_per_tar)
        total.append(rep)
        if rep["num_bad_examples"] > 0:
            bad_tars += 1
            print(json.dumps(rep, ensure_ascii=False), flush=True)
            if args.stop_after_bad_tars > 0 and bad_tars >= args.stop_after_bad_tars:
                break
        if i % 100 == 0:
            print(f"[scan] {i}/{len(shards)} tars checked, bad_tars={bad_tars}", flush=True)

    summary = {
        "num_shards_scanned": len(total),
        "num_bad_tars": sum(1 for x in total if x["num_bad_examples"] > 0),
        "bad_tars": [x for x in total if x["num_bad_examples"] > 0],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
