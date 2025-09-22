#!/bin/bash
#PBS -l walltime=00:05:00
#PBS -A datascience
#PBS -q debug
#PBS -l select=1
#PBS -l filesystems=flare:home
#PBS -N aeris_vis
#PBS -e outputs_inference
#PBS -o outputs_inference
#PBS -W depend=afterany:4095985
#cp -r /flare/Aurora_deployment/vhat/gb25_cli/arrgen/input/mnist /tmp/

cd /flare/Aurora_deployment/vhat/gb25_cli/aeris
source setup_tmp_ds.sh

pip install cartopy

python -m aeris.plotting.rollout \
    --prediction_path=/flare/SAFS/vhat/checkpoints/p_1Bd74b2_1500k_lrrd_base/output/output-1i-180s-24h-50m-10ds-noclamp.hdf5 \
    --variable=sst \
    --name=ls-50m-noclamp \
    --time_start=2020-01-01 \
    --initial=0 \
    --member=0 \
    --ws_y=60 \
    --ws_x=60 \
    --full-format 