#!/bin/bash
################################################################################################
### Orchestrator: RQ1 — Official TSE Format: Zero-Shot vs ICL
###
### Two conditions for each model:
###   zero-shot  (--prompt_format tse_official --num_shots 0):
###     official MCQ format, question + options + query TS, no labeled examples
###   ICL k=1    (--prompt_format tse_official --num_shots 1):
###     same official format + 1 labeled example per class prepended
###
### Four models — spanning text, TS-native, vision, and TS+text modalities:
###   Qwen/Qwen3-8B-Instruct          text-only (TS as numeric array)
###   bytedance-research/ChatTS-8B     TS patch embeddings + text
###   Qwen/Qwen3-VL-8B-Instruct       TS → matplotlib image, VL model
###   anton-hugging/TimeOmni-1-7B     TS as numeric array + reasoning (Qwen2.5 base)
###
### All four use run_single_task_gpu.sh (single RTX 4090, multits env).
###
### Seeds: 3 (0, 3, 6) · Templates: all 98 TSE tids · strategy: random
### Total: 2 conditions × 4 models × 3 seeds × ~98 tids ≈ 2,352 jobs
###
### Prerequisites:
###   python scripts/generate_tse_augmented.py \
###     --tse_repo third_party/TimeSeriesExam \
###     --output qa_dataset_augmented.json \
###     --num_variants 10
###
### Usage:
###   bash experiment_scripts/icl_tse_rq1_official_icl.sh
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/ts-icl"
exp_id="tse_rq1_official_icl"
tse_data="qa_dataset_augmented.json"
batch_size=1
strategy="random"

methods=(
    "Qwen/Qwen3-8B-Instruct"
    "bytedance-research/ChatTS-8B"
    "Qwen/Qwen3-VL-8B-Instruct"
    "anton-hugging/TimeOmni-1-7B"
)
seeds=(0 3 6)
# num_shots: 0 = zero-shot, 1 = ICL (1 example per class)
shots=(0 1)

# Extract all tids dynamically from the augmented dataset
TIDS=$(python -c "
import json
with open('$tse_data') as f:
    data = json.load(f)
tids = sorted(set(e['tid'] for e in data))
print(' '.join(str(t) for t in tids))
")

for k_shots in "${shots[@]}"
do
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
                    --prompt_format "tse_official" \
                    --task_id "icl_tse_${tid}"
            done
        done
    done
done

total=$(echo "$TIDS" | wc -w)
total_jobs=$((${#shots[@]} * ${#methods[@]} * ${#seeds[@]} * total))
echo "Submitted $total_jobs jobs: ${#shots[@]} shot-counts × ${#methods[@]} models × ${#seeds[@]} seeds × $total tids."
