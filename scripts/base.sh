#!/bin/bash

#SBATCH -J base

#SBATCH -A costa.prj
#SBATCH -p gpu_short
#SBATCH --constraint "cascadelake"

#SBATCH -D /well/costa/users/azu506/hccb_proj/hc-cb

#SBATCH --array 1-3
#SBATCH --requeue

# NB you must load the Python module from which your venvs were derived
module load Python/3.11.3-GCCcore-12.3.0

# Activate the architecture-appropriate version of your venv
venv="hccb-proj"
venvs_dir="/well/costa/users/azu506/hccb_proj"
if [ -f "${venvs_dir}/${venv}-${BMRC_GCC_ARCH_NATIVE}/bin/activate" ] ; then
    source "${venvs_dir}/${venv}-${BMRC_GCC_ARCH_NATIVE}/bin/activate"
else
    echo "Failed to identify suitable venv on $(hostname -s): MODULE_CPU_TYPE=${MODULE_CPU_TYPE}; BMRC_GCC_ARCH_NATIVE=${BMRC_GCC_ARCH_NATIVE}" 1>&2
    exit 1
fi

python hippocampus_test.py --experiment_id base_test -hp -c --replicate ${SLURM_ARRAY_TASK_ID}
