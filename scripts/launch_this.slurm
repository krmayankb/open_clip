#!/bin/bash

#SBATCH --account=krishna
#SBATCH --partition=gpu-a100
#SBATCH --job-name=clip
#SBATCH --error=./jobs/mrl_clip/job.err.%j  
#SBATCH --output=./jobs/mrl_clip/job.run.%j
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --gpus=2
#SBATCH --mem-per-gpu=80G
#SBATCH --nodes=1
#SBATCH --mail-user=krmayank@uw.edu
#SBATCH --mail-type=ALL

bash clip.sh