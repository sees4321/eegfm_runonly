# EEG-FM run-only project

This folder is the execution-only refactor derived from your current training recipe.

## What changed
- The 45M uploaded JSON was treated as the reference recipe:
  - `norm_type=layernorm`
  - `full_attn_use_spatial_bias=false`
  - `spatial_qk_type=legendre_anchor`
  - `spec_aux_mode=relational`
- The run-only code removes ablation-only branches and keeps only the parts needed for the final 2-GPU DDP runs.
- `tokens_per_batch` is interpreted as **per-rank valid-token target** for `ShapeBatcher`.
- `tokens_per_update` is interpreted as **global effective target-token budget** per optimizer step, summed across ranks.

## Recommended workflow
1. Run the autotune config first.
2. Update `tokens_per_batch` in the corresponding final train config.
3. Launch the full 2-epoch run.

## Suggested starting points
- 200M-class: encoder 194.955M, total ~214.249M
  - final config starts from `tokens_per_batch=20480` per rank
- 400M-class: encoder 398.188M, total ~423.383M
  - final config starts from `tokens_per_batch=12288` per rank

## Step budget
These final configs use:
- `tokens_per_update = 131072` global effective target tokens
- `max_steps = 45850` for ~2 full-data epochs based on the earlier full-dataset extrapolation

## Commands

### 200M autotune
```bash
accelerate launch --num_processes 2 -m eegfm_runonly.train_autotune_real \
  --model_cfg /path/to/model_200m_encoder195m_total214m.json \
  --train_cfg /path/to/train_200m_2gpu_autotune.json \
  --shards_txt /path/to/full_shards.txt
```

### 200M full run
```bash
accelerate launch --num_processes 2 -m eegfm_runonly.train \
  --model_cfg /path/to/model_200m_encoder195m_total214m.json \
  --train_cfg /path/to/train_200m_2gpu_2epoch.json \
  --shards_txt /path/to/full_shards.txt
```

### 400M autotune
```bash
accelerate launch --num_processes 2 -m eegfm_runonly.train_autotune_real \
  --model_cfg /path/to/model_400m_encoder398m_total423m.json \
  --train_cfg /path/to/train_400m_2gpu_autotune.json \
  --shards_txt /path/to/full_shards.txt
```

### 400M full run
```bash
accelerate launch --num_processes 2 -m eegfm_runonly.train \
  --model_cfg /path/to/model_400m_encoder398m_total423m.json \
  --train_cfg /path/to/train_400m_2gpu_2epoch.json \
  --shards_txt /path/to/full_shards.txt
```
