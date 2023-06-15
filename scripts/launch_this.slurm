#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-a40
#SBATCH --job-name=svd
#SBATCH --error=./jobs/clip_svd/job.err.%j  
#SBATCH --output=./jobs/clip_svd/job.run.%j
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gpus=2
#SBATCH --mem-per-gpu=80G
#SBATCH --nodes=1
#SBATCH --mail-user=krmayank@uw.edu
#SBATCH --mail-type=ALL

bash svd_train.sh