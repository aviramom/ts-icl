# ts-icl: In-Context Learning for Time Series Classification

Evaluating large language models on **few-shot in-context learning (ICL)** for UCR time series
classification. Models receive k labeled time series examples (support set) and a query, and
must predict the query's class — with no gradient updates.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Smoke test with random baseline (no GPU needed)
python run_icl.py \
  --task_id icl_ucr_GunPoint \
  --method random_baseline \
  --num_shots 1 \
  --num_samples 20

# Small LLM (RTX 4090)
python run_icl.py \
  --task_id icl_ucr_GunPoint \
  --method Qwen/Qwen3-4B-Instruct-2507 \
  --num_shots 1 \
  --picking_strategy random \
  --num_samples 50 \
  --random_seed 0
```

## Models

| Method ID | Type | Hardware |
|-----------|------|----------|
| `Qwen/Qwen3-4B-Instruct-2507` | Text LLM | RTX 4090 |
| `Qwen/Qwen3.6-27B` | Large text LLM | 2× RTX 6000 |
| `Qwen/Qwen3.6-27B-image-ts` | Vision LLM (TS→image) | 2× RTX 6000 |
| `Qwen/Qwen3-VL-8B-Instruct` | Vision LLM | RTX 4090 |
| `bytedance-research/ChatTS-8B` | TS-native LLM | RTX 4090 |
| `random_baseline` | Random | CPU |
| `knn_baseline` | 1-NN DTW | CPU |
| `dino_knn_clsa_baseline` | DINOv2 + 1-NN | GPU |

## Datasets

94 UCR time series datasets (fixed-length, context-safe). Task IDs use the prefix `icl_ucr_`,
e.g. `icl_ucr_GunPoint`. Domain descriptions in `ucr_descriptions/`; inject with
`--use_label_desc 1`.

## Full Experiment (SLURM)

```bash
# 94 datasets × 8 seeds × all models (~6,768 jobs)
bash experiment_scripts/icl_ucr_comparison_full.sh

# Description ablation (30 datasets × 5 seeds)
bash experiment_scripts/icl_ucr_desc_comparison.sh
```

## W&B

```bash
python run_icl.py --use_wandb 1 --project aviramom-/ts-icl --exp_id my_run \
  --task_id icl_ucr_GunPoint --method Qwen/Qwen3-4B-Instruct-2507
```

## Citation

[add when published]
