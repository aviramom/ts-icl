#!/bin/bash

################################################################################################
### sbatch configuration parameters (GPU — TimeOmni-1-7B HF smoke test)
### 1×RTX 4090 (24 GB), multits env
###
### Tests three scenarios end-to-end:
###   1. UCR GunPoint 1-shot       — verifies the combined text+TS pipeline
###   2. TSE tid=1 zero-shot       — tse_official format, no examples
###   3. TSE tid=1 1-shot          — tse_official format, 1 labeled example per class
###
### What to look for in the output log:
###   - Model loads without error
###   - "pred:" lines show cleaned text (no <think> or <answer> tags)
###   - "gold:" lines show A/B/C letters
###   - balanced_accuracy is > 0.0 (not all INVALID_PREDICTION)
###   - No "INVALID_PREDICTION" in predicted_answers (check outputs/ JSON)
###
### Prerequisites (TSE runs):
###   qa_dataset_augmented.json must exist in the project root.
###   If missing, generate it first:
###     python scripts/generate_tse_augmented.py \
###       --tse_repo third_party/TimeSeriesExam \
###       --output qa_dataset_augmented.json \
###       --num_variants 10
###
### Submit from project root:
###   sbatch smoke_tests/test_time_omni_smoke.sh
################################################################################################

#SBATCH --partition main
#SBATCH --time 0-00:20:00
#SBATCH --job-name time_omni_smoke
#SBATCH --output logs_terminal/time_omni_smoke_%J.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=60G

### Print debug info ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
echo -e "current path:\t" $PWD "\n\n"

### Start code ###
module load anaconda
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate multits
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Using python from: $(which python)"
python -c "import torch; print('Success! Torch version:', torch.__version__)"

echo ""
echo "=========================================================================="
echo "=== [1/3] UCR GunPoint — 1-shot, combined text+TS pipeline            ==="
echo "=========================================================================="
python run_icl.py \
  --task_id icl_ucr_GunPoint \
  --method anton-hugging/TimeOmni-1-7B \
  --num_shots 1 \
  --picking_strategy random \
  --num_samples 10 \
  --random_seed 0 \
  --batch_size 1 \
  --cache_dir /cs/azencot_fsas/aviramom \
  --data_path /cs/azencot_fsas/multimodal_ts/datasets/ \
  --display_samples 5

echo ""
echo "=========================================================================="
echo "=== [2/3] TSE tid=1 — zero-shot (tse_official, num_shots=0)           ==="
echo "=========================================================================="
python run_icl.py \
  --task_id icl_tse_1 \
  --method anton-hugging/TimeOmni-1-7B \
  --num_shots 0 \
  --prompt_format tse_official \
  --tse_data_path qa_dataset_augmented.json \
  --num_samples 10 \
  --random_seed 0 \
  --batch_size 1 \
  --cache_dir /cs/azencot_fsas/aviramom \
  --display_samples 5

echo ""
echo "=========================================================================="
echo "=== [3/3] TSE tid=1 — 1-shot  (tse_official, num_shots=1)            ==="
echo "=========================================================================="
python run_icl.py \
  --task_id icl_tse_1 \
  --method anton-hugging/TimeOmni-1-7B \
  --num_shots 1 \
  --picking_strategy random \
  --prompt_format tse_official \
  --tse_data_path qa_dataset_augmented.json \
  --num_samples 10 \
  --random_seed 0 \
  --batch_size 1 \
  --cache_dir /cs/azencot_fsas/aviramom \
  --display_samples 5

echo ""
echo "=== TimeOmni smoke test DONE ==="
