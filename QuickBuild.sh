#!/bin/bash
#SBATCH --job-name=comp_apkQ
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=cpu
#SBATCH --account=bdru-delta-cpu
#SBATCH --time=00:10:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module reset

module load cray-hdf5-parallel

export MPICH_GPU_SUPPORT_ENABLED=1

cd build-gpu 
make -j 8
