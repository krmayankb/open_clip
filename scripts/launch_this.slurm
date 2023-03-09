#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-a100
#SBATCH --job-name=tr0.001
#SBATCH --error=./jobs/job.err.%j
#SBATCH --output=./jobs/job.run.%j
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gpus=2
#SBATCH --mem-per-gpu=40G
#SBATCH --nodes=1
#SBATCH --mail-user=krmayank@uw.edu
#SBATCH --mail-type=ALL

bash train.sh 