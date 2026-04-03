#!/bin/bash
#PBS -l min_walltime=03:00:00
#PBS -l max_walltime=06:00:00
#PBS -l walltime=06:00:00
#PBS -A datascience
#PBS -q prod
#PBS -l select=160
#PBS -l filesystems=flare:home
#PBS -N p_1Bd74b4
#PBS -e /flare/SAFS/vhat/outputs
#PBS -o /flare/SAFS/vhat/outputs

cd /flare/Aurora_deployment/vhat/gb25_cli/aeris
#qsub -W depend=afterany:$PBS_JOBID $OUTPUT_PREFIX.sh
source setup_tmp_ds.sh

launch python3 src/aeris/train.py experiment=train
