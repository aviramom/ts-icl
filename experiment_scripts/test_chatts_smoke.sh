#!/bin/bash

################################################################################################
### sbatch configuration parameters (GPU — ChatTS vLLM smoke test)
### 1×RTX 4090 (24 GB), multits env
################################################################################################

#SBATCH --partition main
#SBATCH --time 0-00:20:00
#SBATCH --job-name chatts_vllm_smoke
#SBATCH --output logs_terminal/chatts_vllm_smoke_%J.out
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

python run_icl.py \
  --task_id icl_ucr_GunPoint \
  --method bytedance-research/ChatTS-8B-vllm \
  --num_shots 1 \
  --picking_strategy random \
  --num_samples 20 \
  --random_seed 0 \
  --batch_size 1 \
  --cache_dir /cs/azencot_fsas/aviramom \
  --data_path /cs/azencot_fsas/multimodal_ts/datasets/ \
  --display_samples 3
