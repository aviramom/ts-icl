#!/bin/bash

# Exp: description ablation — 30 UCR datasets that have a description.txt
# Compares Qwen/Qwen3.6-27B-image-ts with and without the domain description.
#   Condition A (use_label_desc=0): no description injected
#   Condition B (use_label_desc=1): full description.txt injected into prompt

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/ts-icl"
exp_id="random_k1_desc_comparison_27b"
batch_size=1
strategy="random"
k_shots=1

seeds=(0 1 2 3 4)

tasks=(
    "icl_ucr_AllGestureWiimoteX"
    "icl_ucr_AllGestureWiimoteY"
    "icl_ucr_AllGestureWiimoteZ"
    "icl_ucr_BME"
    "icl_ucr_Chinatown"
    "icl_ucr_Crop"
    "icl_ucr_DodgersLoopDay"
    "icl_ucr_DodgersLoopGame"
    "icl_ucr_DodgersLoopWeekend"
    "icl_ucr_EthanolLevel"
    "icl_ucr_Fungi"
    "icl_ucr_GunPointAgeSpan"
    "icl_ucr_GunPointMaleVersusFemale"
    "icl_ucr_GunPointOldVersusYoung"
    "icl_ucr_Lightning7"
    "icl_ucr_MelbournePedestrian"
    "icl_ucr_MixedShapesRegularTrain"
    "icl_ucr_MixedShapesSmallTrain"
    "icl_ucr_PickupGestureWiimoteZ"
    "icl_ucr_PowerCons"
    "icl_ucr_Rock"
    "icl_ucr_SemgHandGenderCh2"
    "icl_ucr_SemgHandMovementCh2"
    "icl_ucr_SemgHandSubjectCh2"
    "icl_ucr_ShakeGestureWiimoteZ"
    "icl_ucr_SmoothSubspace"
    "icl_ucr_SwedishLeaf"
    "icl_ucr_SyntheticControl"
    "icl_ucr_TwoPatterns"
    "icl_ucr_UMD"
)

# ── Condition A: no description ───────────────────────────────────────────────
for seed in "${seeds[@]}"
do
    for task in "${tasks[@]}"
    do
        sbatch "$SCRIPT_DIR/run_single_task_gpu_large.sh" \
            --cache_dir "$cache_dir" \
            --method "Qwen/Qwen3.6-27B-image-ts" \
            --display_samples 3 \
            --use_wandb 1 \
            --batch_size "$batch_size" \
            --project "$project" \
            --exp_id "$exp_id" \
            --picking_strategy "$strategy" \
            --num_shots "$k_shots" \
            --num_samples 250 \
            --random_seed "$seed" \
            --use_label_desc 0 \
            --task_id "$task"
    done
done

# ── Condition B: with description ────────────────────────────────────────────
for seed in "${seeds[@]}"
do
    for task in "${tasks[@]}"
    do
        sbatch "$SCRIPT_DIR/run_single_task_gpu_large.sh" \
            --cache_dir "$cache_dir" \
            --method "Qwen/Qwen3.6-27B-image-ts" \
            --display_samples 3 \
            --use_wandb 1 \
            --batch_size "$batch_size" \
            --project "$project" \
            --exp_id "$exp_id" \
            --picking_strategy "$strategy" \
            --num_shots "$k_shots" \
            --num_samples 250 \
            --random_seed "$seed" \
            --use_label_desc 1 \
            --task_id "$task"
    done
done
