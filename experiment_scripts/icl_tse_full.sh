#!/bin/bash
################################################################################################
### Orchestrator: TimeSeriesExam ICL benchmark — all templates × 8 seeds
###
### Run from the project root AFTER generating qa_dataset_augmented.json:
###   python scripts/generate_tse_augmented.py --num_variants 10 --output qa_dataset_augmented.json
###
### Usage:
###   bash experiment_scripts/icl_tse_full.sh
################################################################################################

METHOD=${METHOD:-"Qwen/Qwen3-4B-Instruct-2507"}
NUM_SHOTS=${NUM_SHOTS:-1}
PICKING=${PICKING:-"random"}
USE_DESC=${USE_DESC:-1}
EXP_ID=${EXP_ID:-"tse_icl_full"}
TSE_DATA=${TSE_DATA:-"qa_dataset_augmented.json"}
NUM_VARIANTS=${NUM_VARIANTS:-10}  # must match what was generated

# Extract all tids from the augmented file
TIDS=$(python -c "
import json, sys
with open('$TSE_DATA') as f:
    data = json.load(f)
print(' '.join(str(e['tid']) for e in data))
")

for seed in 0 1 2 3 4 5 6 7; do
  for tid in $TIDS; do
    sbatch experiment_scripts/run_single_task_gpu.sh \
      --task_id icl_tse_${tid} \
      --method ${METHOD} \
      --num_shots ${NUM_SHOTS} \
      --picking_strategy ${PICKING} \
      --use_label_desc ${USE_DESC} \
      --tse_data_path ${TSE_DATA} \
      --random_seed ${seed} \
      --exp_id ${EXP_ID} \
      --use_wandb 1 \
      --project aviramom-/ts-icl \
      --cache_dir /cs/azencot_fsas/aviramom
  done
done

echo "Submitted jobs for $(echo $TIDS | wc -w) templates × 8 seeds with method=${METHOD}"
