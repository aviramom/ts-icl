#!/bin/bash

# Exp: random k=1 — all 94 UCR datasets, Qwen3-VL-8B-Instruct (image-TS)
# 8 seeds × 94 tasks = 752 SLURM jobs, each on a single RTX 4090

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/ts-icl"
exp_id="random_k1_comparison_full"
batch_size=1
strategy="random"
k_shots=1

method="Qwen/Qwen3-VL-8B-Instruct"

seeds=(0 1 2 3 4 5 6 7)

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
    # --- Sensor / Device — first half (20) ---
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
    # --- Sensor / Device — second half (16) ---
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
    # --- Spectrographs / Chemometrics (7) ---
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

for seed in "${seeds[@]}"
do
    for task in "${tasks[@]}"
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
            --num_samples 250 \
            --random_seed "$seed" \
            --task_id "$task"
    done
done

total_jobs=$((${#seeds[@]} * ${#tasks[@]}))
echo "Submitted $total_jobs jobs: $method, ${#seeds[@]} seeds × ${#tasks[@]} tasks."
