#!/bin/bash

################################################################################################
### sbatch configuration parameters (GPU — Qwen3-VL-8B-Thinking vLLM smoke test)
### 1×RTX 4090 (24 GB), multits_large env (has vLLM)
################################################################################################

#SBATCH --partition main
#SBATCH --time 0-00:20:00
#SBATCH --job-name qwenvl8b_thinking_vllm_smoke
#SBATCH --output logs_terminal/qwenvl8b_thinking_vllm_smoke_%J.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=60G

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

echo "Using python from: $(which python)"
python -c "import torch; print('Success! Torch version:', torch.__version__)"

python run_icl.py \
  --task_id icl_ucr_BeetleFly \
  --method Qwen/Qwen3-VL-8B-Thinking-vllm \
  --num_shots 1 \
  --picking_strategy random \
  --num_samples 20 \
  --random_seed 0 \
  --batch_size 1 \
  --max_new_tokens 8192 \
  --thinking_budget 2048 \
  --cache_dir /cs/azencot_fsas/aviramom \
  --data_path /cs/azencot_fsas/multimodal_ts/datasets/ \
  --display_samples 3
