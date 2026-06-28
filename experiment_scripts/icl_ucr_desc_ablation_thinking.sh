#!/bin/bash

# Exp: description ablation — Qwen3-VL-8B with thinking enabled — 94 UCR datasets
# Hypothesis: the thinking model may use domain descriptions more effectively than the
# instruct model (which showed descriptions hurt or gave no benefit in desc_ablation_full).
#
#   Condition A (use_label_desc=0): no description injected into prompt
#   Condition B (use_label_desc=1): description from UCR_DESCRIPTIONS dict in UCRDataset
#
# Backend: vLLM (run_single_task_gpu_thinking.sh — multits_large env on a single RTX 4090).
#   The 8B model fits on one GPU; tensor_parallel_size=1.
# Seeds: 3 (0, 3, 6). num_samples: 150. strategy: random, k=1.
# thinking_budget=2048: soft hint for thinking length (512 caused reasoning to spill into answer).
# max_new_tokens=8192: hard ceiling covering thinking + answer.
# max_seq_length=2048: safe for this dataset set (all ≤12 classes → ≤13 images → ~3,350 tokens).
#   122-dataset set excluded: large-class datasets (FiftyWords/50, ShapesAll/60, Adiac/37 etc.)
#   blow the vLLM max_model_len hard limit and crash the job. 94-dataset set was specifically
#   curated to fit all image-based models.
# Total jobs: 1 model × 2 conditions × 3 seeds × 94 tasks = 564

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/ts-icl"
exp_id="desc_ablation_thinking"
batch_size=1
strategy="random"
k_shots=1
num_samples=150
max_seq_length=2048    # 1-shot input is ~700 tokens; 2048 is safe headroom
max_new_tokens=8192
thinking_budget=2048   # 512 caused premature </think> before model reached a conclusion

methods=( "Qwen/Qwen3-VL-8B-Thinking-vllm" )

seeds=(0 3 6)

tasks=(
    # --- Image / Shape (27) ---
    "icl_ucr_ArrowHead"
    "icl_ucr_BeetleFly"
    "icl_ucr_BirdChicken"
    "icl_ucr_DiatomSizeReduction"
    "icl_ucr_DistalPhalanxOutlineAgeGroup"
    "icl_ucr_DistalPhalanxOutlineCorrect"
    "icl_ucr_DistalPhalanxTW"
    "icl_ucr_FaceAll"
    "icl_ucr_FaceFour"
    "icl_ucr_FacesUCR"
    "icl_ucr_Fish"
    "icl_ucr_Herring"
    "icl_ucr_MedicalImages"
    "icl_ucr_MiddlePhalanxOutlineAgeGroup"
    "icl_ucr_MiddlePhalanxOutlineCorrect"
    "icl_ucr_MiddlePhalanxTW"
    "icl_ucr_OSULeaf"
    "icl_ucr_PhalangesOutlinesCorrect"
    "icl_ucr_ProximalPhalanxOutlineAgeGroup"
    "icl_ucr_ProximalPhalanxOutlineCorrect"
    "icl_ucr_ProximalPhalanxTW"
    "icl_ucr_SwedishLeaf"
    "icl_ucr_Symbols"
    "icl_ucr_Yoga"
    "icl_ucr_Crop"
    "icl_ucr_MixedShapesRegularTrain"
    "icl_ucr_MixedShapesSmallTrain"
    # --- Sensor / Device (36) ---
    "icl_ucr_Car"
    "icl_ucr_ChlorineConcentration"
    "icl_ucr_Computers"
    "icl_ucr_Earthquakes"
    "icl_ucr_ElectricDevices"
    "icl_ucr_FordA"
    "icl_ucr_FordB"
    "icl_ucr_ItalyPowerDemand"
    "icl_ucr_LargeKitchenAppliances"
    "icl_ucr_Lightning2"
    "icl_ucr_Lightning7"
    "icl_ucr_MoteStrain"
    "icl_ucr_Plane"
    "icl_ucr_RefrigerationDevices"
    "icl_ucr_ScreenType"
    "icl_ucr_SmallKitchenAppliances"
    "icl_ucr_SonyAIBORobotSurface1"
    "icl_ucr_SonyAIBORobotSurface2"
    "icl_ucr_StarLightCurves"
    "icl_ucr_Trace"
    "icl_ucr_Wafer"
    "icl_ucr_BME"
    "icl_ucr_Chinatown"
    "icl_ucr_DodgerLoopDay"
    "icl_ucr_DodgerLoopGame"
    "icl_ucr_DodgerLoopWeekend"
    "icl_ucr_FreezerRegularTrain"
    "icl_ucr_FreezerSmallTrain"
    "icl_ucr_HouseTwenty"
    "icl_ucr_InsectEPGRegularTrain"
    "icl_ucr_InsectEPGSmallTrain"
    "icl_ucr_InsectWingbeatSound"
    "icl_ucr_MelbournePedestrian"
    "icl_ucr_PowerCons"
    "icl_ucr_SemgHandGenderCh2"
    "icl_ucr_SmoothSubspace"
    # --- Motion / HAR (16) ---
    "icl_ucr_CricketX"
    "icl_ucr_CricketY"
    "icl_ucr_CricketZ"
    "icl_ucr_GunPoint"
    "icl_ucr_GunPointAgeSpan"
    "icl_ucr_GunPointMaleVersusFemale"
    "icl_ucr_GunPointOldVersusYoung"
    "icl_ucr_ShapeletSim"
    "icl_ucr_ToeSegmentation1"
    "icl_ucr_ToeSegmentation2"
    "icl_ucr_UWaveGestureLibraryX"
    "icl_ucr_UWaveGestureLibraryY"
    "icl_ucr_UWaveGestureLibraryZ"
    "icl_ucr_Worms"
    "icl_ucr_WormsTwoClass"
    "icl_ucr_Fungi"
    # --- ECG / Medical (4) ---
    "icl_ucr_ECG200"
    "icl_ucr_ECG5000"
    "icl_ucr_ECGFiveDays"
    "icl_ucr_TwoLeadECG"
    # --- Spectrographic / Chemometrics (7) ---
    "icl_ucr_Beef"
    "icl_ucr_Coffee"
    "icl_ucr_Ham"
    "icl_ucr_Meat"
    "icl_ucr_OliveOil"
    "icl_ucr_Strawberry"
    "icl_ucr_Wine"
    # --- Simulated / Synthetic (4) ---
    "icl_ucr_CBF"
    "icl_ucr_SyntheticControl"
    "icl_ucr_TwoPatterns"
    "icl_ucr_UMD"
)

# ── Condition A: no description ───────────────────────────────────────────────
for method in "${methods[@]}"
do
    for seed in "${seeds[@]}"
    do
        for task in "${tasks[@]}"
        do
            sbatch "$SCRIPT_DIR/run_single_task_gpu_thinking.sh" \
                --cache_dir "$cache_dir" \
                --method "$method" \
                --display_samples 3 \
                --use_wandb 1 \
                --batch_size "$batch_size" \
                --project "$project" \
                --exp_id "$exp_id" \
                --picking_strategy "$strategy" \
                --num_shots "$k_shots" \
                --num_samples "$num_samples" \
                --max_seq_length "$max_seq_length" \
                --max_new_tokens "$max_new_tokens" \
                --thinking_budget "$thinking_budget" \
                --random_seed "$seed" \
                --use_label_desc 0 \
                --task_id "$task"
        done
    done
done

# ── Condition B: with description (from UCR_DESCRIPTIONS dict in UCRDataset) ─
for method in "${methods[@]}"
do
    for seed in "${seeds[@]}"
    do
        for task in "${tasks[@]}"
        do
            sbatch "$SCRIPT_DIR/run_single_task_gpu_thinking.sh" \
                --cache_dir "$cache_dir" \
                --method "$method" \
                --display_samples 3 \
                --use_wandb 1 \
                --batch_size "$batch_size" \
                --project "$project" \
                --exp_id "$exp_id" \
                --picking_strategy "$strategy" \
                --num_shots "$k_shots" \
                --num_samples "$num_samples" \
                --max_seq_length "$max_seq_length" \
                --max_new_tokens "$max_new_tokens" \
                --thinking_budget "$thinking_budget" \
                --random_seed "$seed" \
                --use_label_desc 1 \
                --task_id "$task"
        done
    done
done

total_tasks=${#tasks[@]}
total_jobs=$((${#methods[@]} * 2 * ${#seeds[@]} * total_tasks))
echo "Submitted $total_jobs jobs: ${#methods[@]} model × 2 conditions × ${#seeds[@]} seeds × $total_tasks tasks."
