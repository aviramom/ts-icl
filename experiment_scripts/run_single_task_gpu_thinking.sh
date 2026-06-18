#!/bin/bash

################################################################################################
### sbatch configuration parameters (GPU — single task/seed job, vLLM small models)
### Uses multits_large env (has vLLM) on a single RTX 4090.
### For 8B vLLM models (Qwen3-VL-8B-Thinking-vllm, ChatTS-8B-vllm, etc.)
### that need vLLM but do NOT require the 2×RTX 6000 reserved for 27B models.
################################################################################################

#SBATCH --partition main
#SBATCH --time 0-01:00:00
#SBATCH --job-name icl_single
#SBATCH --output logs_terminal/icl_single_%J.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --exclude=ise-4090-02
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

python run_icl.py "$@"
