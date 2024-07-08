#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -S 0
 
set -x
export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
 
export OMP_NUM_THREADS=7
export HYDRAGNN_AGGR_BACKEND=mpi

source /lustre/orion/cph161/world-shared/mlupopa/module-to-load-frontier-rocm571.sh
source /lustre/orion/cph161/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
conda activate hydragnn_rocm571
 
export PYTHONPATH=/lustre/orion/cph161/world-shared/mlupopa/ADIOS_frontier_rocm571/install/lib/python3.9/site-packages/:$PYTHONPATH
 
export PYTHONPATH=$PWD:$PYTHONPATH
 
cd examples/LennardJones/
 
srun -n$((SLURM_JOB_NUM_NODES*8)) --gpus-per-task=1 --gpu-bind=closest python -u mae_analysis.py