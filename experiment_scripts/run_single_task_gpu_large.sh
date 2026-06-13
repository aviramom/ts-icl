#!/bin/bash

################################################################################################
### sbatch configuration parameters (GPU — single task/seed job, large models)
### Uses multits_large env (newer transformers for Qwen3.6-27B / qwen3_5 architecture)
### 2×RTX 6000 (96 GB) for 27B models
################################################################################################

#SBATCH --partition main
#SBATCH --time 0-01:30:00
#SBATCH --job-name icl_single_large
#SBATCH --output logs_terminal/icl_single_large_%J.out
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
# Prepend conda's lib so its libstdc++/libzmq take priority over the old system versions
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
# Disable NCCL P2P — required when GPUs are on different PCIe switches (non-adjacent indices)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "Using python from: $(which python)"
python -c "import torch; print('Success! Torch version:', torch.__version__)"

python run_icl.py "$@"


