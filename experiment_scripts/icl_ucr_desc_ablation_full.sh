#!/bin/bash

# Exp: description ablation — all 128 UCR datasets
# Compares ChatTS-8B and Qwen3-VL-8B with and without domain descriptions.
#   Condition A (use_label_desc=0): no description injected into prompt
#   Condition B (use_label_desc=1): description from UCR_DESCRIPTIONS dict in UCRDataset
#
# Both models run on a single RTX 4090 (multits env, run_single_task_gpu.sh).
# Seeds: 3 (0, 3, 6). num_samples: 150. strategy: random, k=1.
# Total jobs: 2 models × 2 conditions × 3 seeds × 122 tasks ≈ 1,464
# (6 variable-length datasets are commented out below — they cannot be loaded as fixed-length ARFF)

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/ts-icl"
exp_id="desc_ablation_full"
batch_size=1
strategy="random"
k_shots=1
num_samples=150

methods=( "bytedance-research/ChatTS-8B" "Qwen/Qwen3-VL-8B-Instruct" )

seeds=(0 3 6)

tasks=(
    # --- Image / Shape ---
    "icl_ucr_Adiac"
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
    "icl_ucr_FiftyWords"
    "icl_ucr_Fish"
    "icl_ucr_HandOutlines"
    "icl_ucr_Herring"
    "icl_ucr_MedicalImages"
    "icl_ucr_MiddlePhalanxOutlineAgeGroup"
    "icl_ucr_MiddlePhalanxOutlineCorrect"
    "icl_ucr_MiddlePhalanxTW"
    "icl_ucr_MixedShapesRegularTrain"
    "icl_ucr_MixedShapesSmallTrain"
    "icl_ucr_OSULeaf"
    "icl_ucr_PhalangesOutlinesCorrect"
    "icl_ucr_ProximalPhalanxOutlineAgeGroup"
    "icl_ucr_ProximalPhalanxOutlineCorrect"
    "icl_ucr_ProximalPhalanxTW"
    "icl_ucr_ShapesAll"
    "icl_ucr_SwedishLeaf"
    "icl_ucr_Symbols"
    "icl_ucr_WordSynonyms"
    "icl_ucr_Yoga"
    "icl_ucr_Crop"

    # --- Sensor / Device ---
    "icl_ucr_ACSF1"
    "icl_ucr_BME"
    "icl_ucr_Car"
    "icl_ucr_Chinatown"
    "icl_ucr_ChlorineConcentration"
    "icl_ucr_Computers"
    "icl_ucr_DodgerLoopDay"
    "icl_ucr_DodgerLoopGame"
    "icl_ucr_DodgerLoopWeekend"
    "icl_ucr_Earthquakes"
    "icl_ucr_ElectricDevices"
    "icl_ucr_EthanolLevel"
    "icl_ucr_FordA"
    "icl_ucr_FordB"
    "icl_ucr_FreezerRegularTrain"
    "icl_ucr_FreezerSmallTrain"
    "icl_ucr_HouseTwenty"
    "icl_ucr_InsectEPGRegularTrain"
    "icl_ucr_InsectEPGSmallTrain"
    "icl_ucr_InsectWingbeatSound"
    "icl_ucr_ItalyPowerDemand"
    "icl_ucr_LargeKitchenAppliances"
    "icl_ucr_Lightning2"
    "icl_ucr_Lightning7"
    "icl_ucr_MelbournePedestrian"
    "icl_ucr_MoteStrain"
    "icl_ucr_Plane"
    "icl_ucr_PowerCons"
    "icl_ucr_RefrigerationDevices"
    "icl_ucr_Rock"
    "icl_ucr_ScreenType"
    "icl_ucr_SemgHandGenderCh2"
    "icl_ucr_SemgHandMovementCh2"
    "icl_ucr_SemgHandSubjectCh2"
    "icl_ucr_SmallKitchenAppliances"
    "icl_ucr_SmoothSubspace"
    "icl_ucr_SonyAIBORobotSurface1"
    "icl_ucr_SonyAIBORobotSurface2"
    "icl_ucr_StarLightCurves"
    "icl_ucr_Trace"
    "icl_ucr_Wafer"

    # --- Motion / HAR ---
    "icl_ucr_AllGestureWiimoteX"
    "icl_ucr_AllGestureWiimoteY"
    "icl_ucr_AllGestureWiimoteZ"
    "icl_ucr_CricketX"
    "icl_ucr_CricketY"
    "icl_ucr_CricketZ"
    "icl_ucr_Fungi"
    "icl_ucr_GunPoint"
    "icl_ucr_GunPointAgeSpan"
    "icl_ucr_GunPointMaleVersusFemale"
    "icl_ucr_GunPointOldVersusYoung"
    "icl_ucr_Haptics"
    "icl_ucr_InlineSkate"
    "icl_ucr_PickupGestureWiimoteZ"
    "icl_ucr_ShakeGestureWiimoteZ"
    "icl_ucr_ShapeletSim"
    "icl_ucr_ToeSegmentation1"
    "icl_ucr_ToeSegmentation2"
    "icl_ucr_UWaveGestureLibraryAll"
    "icl_ucr_UWaveGestureLibraryX"
    "icl_ucr_UWaveGestureLibraryY"
    "icl_ucr_UWaveGestureLibraryZ"
    "icl_ucr_Worms"
    "icl_ucr_WormsTwoClass"
    # Variable-length (cannot load as fixed-length ARFF — skipped):
    # "icl_ucr_GestureMidAirD1"
    # "icl_ucr_GestureMidAirD2"
    # "icl_ucr_GestureMidAirD3"
    # "icl_ucr_GesturePebbleZ1"
    # "icl_ucr_GesturePebbleZ2"

    # --- ECG / Medical ---
    "icl_ucr_CinCECGTorso"
    "icl_ucr_ECG200"
    "icl_ucr_ECG5000"
    "icl_ucr_ECGFiveDays"
    "icl_ucr_EOGHorizontalSignal"
    "icl_ucr_EOGVerticalSignal"
    "icl_ucr_NonInvasiveFetalECGThorax1"
    "icl_ucr_NonInvasiveFetalECGThorax2"
    "icl_ucr_PigAirwayPressure"
    "icl_ucr_PigArtPressure"
    "icl_ucr_PigCVP"
    "icl_ucr_TwoLeadECG"

    # --- Spectrographic / Chemometrics ---
    "icl_ucr_Beef"
    "icl_ucr_Coffee"
    "icl_ucr_Ham"
    "icl_ucr_Meat"
    "icl_ucr_OliveOil"
    "icl_ucr_Strawberry"
    "icl_ucr_Wine"

    # --- Simulated / Synthetic ---
    "icl_ucr_CBF"
    "icl_ucr_Mallat"
    "icl_ucr_Phoneme"
    "icl_ucr_SyntheticControl"
    "icl_ucr_TwoPatterns"
    "icl_ucr_UMD"
    # Variable-length (cannot load as fixed-length ARFF — skipped):
    # "icl_ucr_PLAID"
)

# ── Condition A: no description ───────────────────────────────────────────────
for method in "${methods[@]}"
do
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
                --num_samples "$num_samples" \
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
                --num_samples "$num_samples" \
                --random_seed "$seed" \
                --use_label_desc 1 \
                --task_id "$task"
        done
    done
done

total_tasks=${#tasks[@]}
total_jobs=$((${#methods[@]} * 2 * ${#seeds[@]} * total_tasks))
echo "Submitted $total_jobs jobs: ${#methods[@]} models × 2 conditions × ${#seeds[@]} seeds × $total_tasks tasks."
