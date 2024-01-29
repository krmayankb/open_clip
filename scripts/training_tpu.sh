#!/bin/bash

export PJRT_DEVICE=TPU
export XLA_IR_DEBUG=1
export XLA_METRICS_FILE=1
#sudo kill -9 $(sudo lsof -w /dev/accel0 | awk 'NR>1 {print $2}' | uniq)

python -c "import os; os.environ.pop('LD_PRELOAD', None)"


cd ~/open_clip/src/ 
python3 -m training.main \
    --model "ViT-B-32" \
    --train-data "/home/krmayank/data/medium/data/{00000000..00001919}.tar" \
    --imagenet-val="/home/krmayank/data/imagenet/imagenet_val/" \
    --dataset-type webdataset \
    --precision amp \
    --gather-with-grad \
    --local-loss \
    --force_mrl_loss \
    --mrl_loss_weights "1,1,1,1,1" \
    --mrl_dim_to_consider "768,384,192,96,48" \
    --batch-size 128 \
    --accum-freq 1 \
    --workers 4 \
    --epochs 3 \
    --warmup 4 \
    --zeroshot-frequency 1 \
    --seed 1234 \
    --gather-with-grad \
    --train-num-samples 10000 \
    --val-num-samples 10000
#    --report-to 'wandb' \
#    --wandb-project-name "mrl_clip_training" 
# --val-data "/home/krmayank/data/medium/data/{00000000..00000000}.tar" \
# --imagenet-val="/home/krmayank/data/imagenet/imagenet_val/" \
