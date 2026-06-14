#!/bin/bash
################################################################################################
### Smoke test: TimeSeriesExam ICL — quick end-to-end check (no SLURM needed)
###
### Prerequisites:
###   1. Generate augmented data (small, 3 variants for speed):
###      python scripts/generate_tse_augmented.py \
###          --num_variants 3 --ts_length 256 --output qa_dataset_augmented.json --tids 1,3,65
###   2. Run this script from the project root.
################################################################################################

set -e

echo "=== TSE smoke test: random_baseline, tid=1, 1-shot ==="
python run_icl.py \
  --task_id icl_tse_1 \
  --method random_baseline \
  --num_shots 1 \
  --picking_strategy random \
  --use_label_desc 1 \
  --tse_data_path qa_dataset_augmented.json \
  --tse_test_fraction 0.4 \
  --random_seed 0 \
  --display_samples 2

echo ""
echo "=== TSE smoke test: random_baseline, tid=65 (anomaly detection) ==="
python run_icl.py \
  --task_id icl_tse_65 \
  --method random_baseline \
  --num_shots 1 \
  --picking_strategy random \
  --use_label_desc 1 \
  --tse_data_path qa_dataset_augmented.json \
  --random_seed 0 \
  --display_samples 2

echo ""
echo "=== TSE smoke test DONE ==="
