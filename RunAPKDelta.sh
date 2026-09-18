#!/bin/bash
#SBATCH --job-name="comp_apk"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bdru-delta-gpu   # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 01:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module reset


module load cray-hdf5-parallel
export MPICH_GPU_SUPPORT_ENABLED=1

srun ./athenaPK -i turbulence.in

