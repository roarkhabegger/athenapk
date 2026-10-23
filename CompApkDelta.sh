#!/bin/bash
#SBATCH --mem=64g  #<- 256GB per node, 128 cores per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32  # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu
#SBATCH --account=bdru-delta-cpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=comp_apk
#SBATCH --time=00:05:00      # hh:mm:ss for the job
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

rm -rf build-gpu
module reset


module load cray-hdf5-parallel

export MPICH_GPU_SUPPORT_ENABLED=1

cmake -S . -B build-gpu \
  -DCMAKE_CXX_COMPILER=CC \
  -DMPI_CXX_COMPILER=CC \
  -DKokkos_ARCH_ZEN3=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DPARTHENON_DISABLE_OPENPMD=ON \
  -DHDF5_ROOT="/opt/cray/pe/hdf5-parallel/1.14.3.9/gnu/12.2" \
  -DAthenaPK_ENABLE_TESTING=OFF

cd build-gpu 
make -j 16
