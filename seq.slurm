#!/bin/bash
#SBATCH -p gpus
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --job-name=cuda-test

MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

gcc ./src/cpu.c -o ./bin/seq && srun nsys profile ./bin/seq

wait
