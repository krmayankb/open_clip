cd /mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src
torchrun --nproc_per_node 2 --master_port 3467 -m training.main \
    --model "ViT-B-16" \
    --train-data "/mmfs1/data/yfcc-tmp/cc_3m/train_shards/shard_{000000..003318}.tar" \
    --imagenet-val "/mmfs1/data/yfcc-tmp/imagenet/val/" \
    --dataset-type webdataset \
    --precision amp \
    --gather-with-grad \
    --local-loss \
    --batch-size 256 \
    --accum-freq 2 \
    --workers 6 \
    --epochs 40 \
    --warmup 2000 \
    --zeroshot-frequency 2 \
    --seed 0 \
    --report-to 'wandb' \
    --wandb-project-name "byol_clip_training" \
    --logs "/mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src/logs/byol_clip" \
    --name "clip_original_b256_accum_2_ep40_final_V3" 
