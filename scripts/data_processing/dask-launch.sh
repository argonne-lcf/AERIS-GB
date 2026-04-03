#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -A datascience
#PBS -q prod
#PBS -l select=1
#PBS -l filesystems=flare:home
#PBS -N aeris_process
#PBS -e outputs_inference
#PBS -o outputs_inference
#PBS -W depend=afterany:4569053

set -e 
cleanup() {
    echo "Cleaning up Dask cluster..."
    kill "$SCHEDULER_PID" 2>/dev/null || true
    kill "$WORKERS_PID" 2>/dev/null || true
    rm -f scheduler.json
}
trap cleanup SIGINT EXIT

#ssh -L 8787:127.0.0.1:8787 -J vhat@aurora.alcf.anl.gov vhat@x4604c7s0b0n0
#module load frameworks
#source /flare/Aurora_deployment/vhat/gb25_cli/aeris/src/aeris/data/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/activate
cd /flare/Aurora_deployment/vhat/gb25_cli/aeris/src/aeris/data
PBS_O_WORKDIR=$(pwd) source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh)
#VENV_DIR=/flare/Aurora_deployment/vhat/gb25_cli/aeris/src/aeris/data/venvs/aurora_nre_models_frameworks-2024.2.1_u1
PBS_O_WORKDIR=$(pwd) ezpz_setup_env
export NUMEXPR_MAX_THREADS=512 # annoying pandas warning
DASHBOARD_IP=$(hostname -I | awk '{print $1}')
DASHBOARD_ADDRESS="${DASHBOARD_IP}:8787"
echo "Dask dashboard address: ${DASHBOARD_ADDRESS}"

mpiexec --verbose --envall -n 1 --hostfile ${PBS_NODEFILE} \
    dask-mpi --scheduler-file scheduler.json --dashboard-address ${DASHBOARD_ADDRESS} &
SCHEDULER_PID=$!

echo "Waiting for scheduler to start..."
while [ ! -f scheduler.json ]; do
    sleep 1
done
echo "Dask scheduler initialized, scheduler.json found."

NUM_WORKERS_PER_NODE=24

mpiexec --verbose --envall -n $(( NHOSTS * NUM_WORKERS_PER_NODE )) -ppn ${NUM_WORKERS_PER_NODE} \
    --hostfile ${PBS_NODEFILE} --cpu-bind depth -d 8 \
    dask-mpi --scheduler-file scheduler.json --no-scheduler --nthreads 4 \
    --dashboard-address ${DASHBOARD_ADDRESS} &
WORKERS_PID=$!

sleep 30
#/flare/Aurora_deployment/vhat/gb25_cli/aeris/checkpoints/p_1Bf2w3m/output/
#/flare/Aurora_deployment/vhat/gb25_cli/aeris/checkpoints/p_1Bf2w/output/
echo "Starting Python job..."
python -m aeris.data.bh52zarr_dist \
    --dask-scheduler scheduler.json \
    -i /flare/Aurora_deployment/vhat/gb25_cli/aeris/checkpoints/p_xB_1331k/output/output-55i-56s-162h.hdf5 \
    -o /flare/Aurora_deployment/vhat/gb25_cli/aeris/eval_out \
    -n p_xB_1331k-output-55i-56s-162h \
    --interval 162

echo "Python job completed."