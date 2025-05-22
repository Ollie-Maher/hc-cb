#!/bin/bash

#SBATCH -J HPC_CB

#SBATCH -A costa.prj
#SBATCH -p gpu_short
#SBATCH --constraint "a100|rtx8000"

#SBATCH -D /well/costa/users/azu506/hccb_proj

#SBATCH --array=0-2
#SBATCH --requeue

# NB you must load the Python module from which your venvs were derived
module load Python-bundle/3.11.3-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# Activate the venv
source "/well/costa/users/azu506/hccb_proj/hccb-proj-cascadelake/bin/activate"


python ./hc-cb/hippocampus_test.py --experiment_id fixed_CB_and_HPC --replicate ${SLURM_ARRAY_TASK_ID}