#!/bin/bash
################################################################################################
### Orchestrator: TSE question-text ablation — all templates × 3 seeds
###
### Compares ChatTS-8B and Qwen3-VL-8B with and without the question/option legend.
###   Condition A (use_label_desc=0): model sees only labeled TS examples, no question text
###   Condition B (use_label_desc=1): question + option legend prepended to the prompt
###
### Both models run on a single RTX 4090 (multits env, run_single_task_gpu.sh).
### Seeds: 3 (0, 3, 6). strategy: random, k=1.
### Total jobs: 2 models × 2 conditions × 3 seeds × 98 tids ≈ 1,176
###
### Prerequisites:
###   python scripts/generate_tse_augmented.py \
###     --tse_repo third_party/TimeSeriesExam \
###     --output qa_dataset_augmented.json \
###     --num_variants 10
###
### Usage:
###   bash experiment_scripts/icl_tse_desc_ablation_full.sh
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/ts-icl"
exp_id="tse_desc_ablation_full"
tse_data="qa_dataset_augmented.json"
batch_size=1
strategy="random"
k_shots=1

methods=( "bytedance-research/ChatTS-8B" "Qwen/Qwen3-VL-8B-Instruct" )

seeds=(0 3 6)

# Extract all tids dynamically from the augmented dataset
TIDS=$(python -c "
import json
with open('$tse_data') as f:
    data = json.load(f)
tids = sorted(set(e['tid'] for e in data))
print(' '.join(str(t) for t in tids))
")

# ── Condition A: no question text ─────────────────────────────────────────────
for method in "${methods[@]}"
do
    for seed in "${seeds[@]}"
    do
        for tid in $TIDS
        do
            sbatch "$SCRIPT_DIR/run_single_task_gpu.sh" \
                --cache_dir "$cache_dir" \
                --method "$method" \
                --display_samples 3 \
                --use_wandb 1 \
                --batch_size "$batch_size" \
                --project "$project" \
                --exp_id "$exp_id" \
                --picking_strategy "$strategy" \
                --num_shots "$k_shots" \
                --random_seed "$seed" \
                --tse_data_path "$tse_data" \
                --use_label_desc 0 \
                --task_id "icl_tse_${tid}"
        done
    done
done

# ── Condition B: with question + option legend ────────────────────────────────
for method in "${methods[@]}"
do
    for seed in "${seeds[@]}"
    do
        for tid in $TIDS
        do
            sbatch "$SCRIPT_DIR/run_single_task_gpu.sh" \
                --cache_dir "$cache_dir" \
                --method "$method" \
                --display_samples 3 \
                --use_wandb 1 \
                --batch_size "$batch_size" \
                --project "$project" \
                --exp_id "$exp_id" \
                --picking_strategy "$strategy" \
                --num_shots "$k_shots" \
                --random_seed "$seed" \
                --tse_data_path "$tse_data" \
                --use_label_desc 1 \
                --task_id "icl_tse_${tid}"
        done
    done
done

total=$(echo "$TIDS" | wc -w)
total_jobs=$((${#methods[@]} * 2 * ${#seeds[@]} * total))
echo "Submitted $total_jobs jobs: ${#methods[@]} models × 2 conditions × ${#seeds[@]} seeds × $total tids."
