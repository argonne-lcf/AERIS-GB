#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -A datascience
#PBS -q prod
#PBS -l select=16
#PBS -l filesystems=flare:home
#PBS -N process_wp_data

cd /flare/Aurora_deployment/vhat/gb25_cli/aeris

module load frameworks
source venvs/aeris_2025_5/bin/activate

export PYTHONUSERBASE=/flare/Aurora_deployment/vhat/gb25_cli/pythonuserbase

export CCL_WORKER_AFFINITY=1,9,17,25,33,41,53,61,69,77,85,93
export CPU_BIND="list:2-8:10-16:18-24:26-32:34-40:42-48:54-60:62-68:70-76:78-84:86-92:94-100"
export NUMEXPR_MAX_THREADS=7
export OMP_NUM_THREADS=7

export HDF5_USE_FILE_LOCKING=FALSE

ulimit -S -s unlimited

ulimit -n 524288

#export CPU_BIND="list:0-51"
cd /flare/Aurora_deployment/vhat/gb25_cli/aeris/src/aeris/data
mpiexec --pmi=pmix -l --line-buffer \
-n 192 --ppn 12 --cpu-bind $CPU_BIND python wp_create.py
