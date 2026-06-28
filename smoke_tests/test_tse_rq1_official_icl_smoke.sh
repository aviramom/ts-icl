#!/bin/bash
################################################################################################
### Smoke test: TSE RQ1 — Official ICL format, zero-shot vs k=1
### Tests both conditions (num_shots 0 and 1) with random_baseline (no GPU required).
###
### Purpose: verify that --prompt_format tse_official builds correct prompts for both
### zero-shot and ICL conditions, label extraction works (A/B/C options), and the
### pipeline runs end-to-end without crashes.
###
### Prerequisites:
###   1. Generate augmented data (3 variants for speed, tids 1 and 65 only):
###      python scripts/generate_tse_augmented.py \
###          --num_variants 3 --output qa_dataset_augmented.json --tids 1,65
###   2. Run from the project root (no SLURM needed):
###      bash smoke_tests/test_tse_rq1_official_icl_smoke.sh
###
### Expected output:
###   - Zero-shot: prompt has question + options + "Time Series:" + format_hint, NO examples block
###   - ICL k=1:   prompt has "Here are some labeled examples:" + "Answer: A) ..." block first
###   - balanced_accuracy ~0.33 for 3-class (tid=1), ~0.5 for 2-class (tid=65)
###   - INVALID_PREDICTION count should be 0 (all predictions are valid A/B/C letters)
################################################################################################

set -e

echo "========================================================"
echo "  TSE RQ1 Official ICL Smoke Test"
echo "========================================================"

echo ""
echo "--- [1/4] Zero-Shot (num_shots=0), tid=1 (trend type, 3 classes) ---"
python run_icl.py \
  --task_id icl_tse_1 \
  --method random_baseline \
  --prompt_format tse_official \
  --num_shots 0 \
  --picking_strategy random \
  --tse_data_path qa_dataset_augmented.json \
  --tse_test_fraction 0.4 \
  --random_seed 0 \
  --display_samples 2

echo ""
echo "--- [2/4] ICL k=1 (num_shots=1), tid=1 (trend type, 3 classes) ---"
python run_icl.py \
  --task_id icl_tse_1 \
  --method random_baseline \
  --prompt_format tse_official \
  --num_shots 1 \
  --picking_strategy random \
  --tse_data_path qa_dataset_augmented.json \
  --tse_test_fraction 0.4 \
  --random_seed 0 \
  --display_samples 2

echo ""
echo "--- [3/4] Zero-Shot (num_shots=0), tid=65 (anomaly detection) ---"
python run_icl.py \
  --task_id icl_tse_65 \
  --method random_baseline \
  --prompt_format tse_official \
  --num_shots 0 \
  --picking_strategy random \
  --tse_data_path qa_dataset_augmented.json \
  --random_seed 0 \
  --display_samples 1

echo ""
echo "--- [4/4] ICL k=1 (num_shots=1), tid=65 (anomaly detection) ---"
python run_icl.py \
  --task_id icl_tse_65 \
  --method random_baseline \
  --prompt_format tse_official \
  --num_shots 1 \
  --picking_strategy random \
  --tse_data_path qa_dataset_augmented.json \
  --random_seed 0 \
  --display_samples 1

echo ""
echo "========================================================"
echo "  Smoke test DONE — check prompts above:"
echo "  Zero-shot: no 'Example N Time Series' lines"
echo "  ICL k=1:   has 'Here are some labeled examples:' block"
echo "             labels formatted as 'Answer: A) <option text>'"
echo "========================================================"
