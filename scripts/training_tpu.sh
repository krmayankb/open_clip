#!/bin/bash

export PJRT_DEVICE=TPU
export XLA_IR_DEBUG=1
export XLA_METRICS_FILE=1
#sudo kill -9 $(sudo lsof -w /dev/accel0 | awk 'NR>1 {print $2}' | uniq)

python -c "import os; os.environ.pop('LD_PRELOAD', None)"


cd ~/open_clip/src/ 

python3 -m training.main \
    --model "ViT-B-16" \
    --train-data "/home/krmayank/data/medium/data/{00000000..00001919}.tar" \
    --imagenet-val="/home/krmayank/data/imagenet/imagenet_val/" \
    --precision "amp_bfloat16" \
    --dataset-type webdataset \
    --use_tpu \
    --force_mrl_loss \
    --mrl_dim_to_consider "768,384,192,96,48" \
    --gather-with-grad \
    --local-loss \
    --batch-size 64 \
    --accum-freq 1 \
    --workers 2 \
    --epochs 3 \
    --warmup 1 \
    --zeroshot-frequency 1 \
    --seed 0 \
    --logs "../logs/" \
    --train-num-samples 10000 \
    --val-num-samples 10000 \
#    --report-to 'wandb' \
#    --wandb-project-name "mrl_clip_training" 
# --val-data "/home/krmayank/data/medium/data/{00000000..00000000}.tar" \
# --imagenet-val="/home/krmayank/data/imagenet/imagenet_val/" \
