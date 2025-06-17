#!/bin/bash
#SBATCH -p gpus
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --job-name=cublasxt-test

# MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
# MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

nvcc ./src/multigpu-cublas.cu -o ./bin/cublas -lcublas && sun ./bin/cublas

wait
