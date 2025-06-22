#!/bin/bash
#SBATCH -p gpus
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --output=output.log
#SBATCH --error=error.err
#SBATCH --job-name=test

srun nsys profile ./bin/main

wait
