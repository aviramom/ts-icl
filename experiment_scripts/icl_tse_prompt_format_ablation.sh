#!/bin/bash
################################################################################################
### Orchestrator: TSE prompt-format ablation — QwenVL-8B-Instruct, 4 conditions
###
### Tests four ways of structuring the prompt for TSE:
###   no_support  — zero-shot: question + query only, no labeled examples
###   desc_first  — question prepended before examples (existing condition B)
###   no_desc     — examples only, no question text (existing condition A)
###   desc_last   — examples first, question immediately before the query (recency-bias test)
###
### Model: Qwen/Qwen3-VL-8B-Instruct. Single RTX 4090 (multits env).
### Seeds: 3 (0, 3, 6). Strategy: random, k=1.
### Total jobs: 4 formats × 1 model × 3 seeds × ~98 tids ≈ 1,176
###
### Prerequisites:
###   python scripts/generate_tse_augmented.py \
###     --tse_repo third_party/TimeSeriesExam \
###     --output qa_dataset_augmented.json \
###     --num_variants 10
###
### Usage:
###   bash experiment_scripts/icl_tse_prompt_format_ablation.sh
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/ts-icl"
exp_id="tse_prompt_format_ablation_qwenvl"
tse_data="qa_dataset_augmented.json"
batch_size=1
strategy="random"
k_shots=1

methods=("Qwen/Qwen3-VL-8B-Instruct")
seeds=(0 3 6)
formats=("no_support" "desc_first" "no_desc" "desc_last")

# Extract all tids dynamically from the augmented dataset
TIDS=$(python -c "
import json
with open('$tse_data') as f:
    data = json.load(f)
tids = sorted(set(e['tid'] for e in data))
print(' '.join(str(t) for t in tids))
")

for fmt in "${formats[@]}"
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
                    --prompt_format "$fmt" \
                    --task_id "icl_tse_${tid}"
            done
        done
    done
done

total=$(echo "$TIDS" | wc -w)
total_jobs=$((${#formats[@]} * ${#methods[@]} * ${#seeds[@]} * total))
echo "Submitted $total_jobs jobs: ${#formats[@]} formats × ${#methods[@]} models × ${#seeds[@]} seeds × $total tids."
