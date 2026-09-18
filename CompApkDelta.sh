#!/bin/bash
#SBATCH --job-name="comp_apk"
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu  # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bdru-delta-cpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --time=00:30:00      # hh:mm:ss for the job
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

rm -rf build-gpu
module reset
# module load nvhpc

# module load cray-mpich

module load cray-hdf5-parallel

export MPICH_GPU_SUPPORT_ENABLED=1
# export NVHPC_CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda"
# export NVCOMPILER_COMM_LIBS_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/comm_libs"
# HDF5_ROOT="/opt/cray/pe/hdf5-parallel/1.14.3.9/gnu/12.2"
# MPI_ROOT="$NVCOMPILER_COMM_LIBS_HOME/mpi"
cmake -S . -B build-gpu \
  -DCMAKE_CXX_COMPILER=CC \
  -DMPI_CXX_COMPILER=CC \
  -DKokkos_ARCH_ZEN3=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DPARTHENON_DISABLE_OPENPMD=ON \
  -DHDF5_ROOT="/opt/cray/pe/hdf5-parallel/1.14.3.9/gnu/12.2" \
  -DAthenaPK_ENABLE_TESTING=OFF
# mpicxx -show
# cmake -S. -Bbuild-gpu -DKokkos_ARCH_ZEN3=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
#         -DMPI_ROOT="$MPI_ROOT" \
#         -DMPI_CXX_COMPILER="$MPI_ROOT/bin/mpicxx" \
#         -DHDF5_ROOT="$HDF5_ROOT"

cd build-gpu 
make -j 16
