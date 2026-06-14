#!/bin/bash

################################################################################################
### sbatch configuration parameters (GPU — Qwen3.6-27B image-ts vLLM smoke test)
### 2×RTX 6000 (96 GB), multits_large env
################################################################################################

#SBATCH --partition main
#SBATCH --time 0-00:30:00
#SBATCH --job-name qwen27b_image_smoke
#SBATCH --output logs_terminal/qwen27b_image_smoke_%J.out
#SBATCH --gpus=rtx_6000:2
#SBATCH --exclude=ee-l40s-01
#SBATCH --mem=128G

### Print debug info ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
echo -e "current path:\t" $PWD "\n\n"

### Start code ###
module load anaconda
module load cuda/12.4
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate multits_large
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "Using python from: $(which python)"
python -c "import torch; print('Success! Torch version:', torch.__version__)"

python run_icl.py \
  --task_id icl_ucr_GunPoint \
  --method Qwen/Qwen3.6-27B-image-ts \
  --num_shots 1 \
  --picking_strategy random \
  --num_samples 20 \
  --random_seed 0 \
  --batch_size 1 \
  --cache_dir /cs/azencot_fsas/aviramom \
  --data_path /cs/azencot_fsas/multimodal_ts/datasets/ \
  --display_samples 3
