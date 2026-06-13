#!/bin/bash

################################################################################################
### sbatch configuration parameters (CPU-only — baselines, single task/seed job)
################################################################################################

#SBATCH --partition main
#SBATCH --time 0-00:30:00
#SBATCH --job-name icl_baseline_single
#SBATCH --output logs_terminal/icl_baseline_single_%J.out
#SBATCH --mem=16G

### Print debug info ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
echo -e "current path:\t" $PWD "\n\n"

### Start code ###
module load anaconda
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate multits

echo "Using python from: $(which python)"
python run_icl.py "$@"
