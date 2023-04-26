cd /mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src
torchrun --nproc_per_node 2 --master_port 1235 -m training.main \
    --model "ViT-B-16" \
    --train-data "/mmfs1/data/yfcc-tmp/cc_3m/train_shards/shard_{000000..003318}.tar" \
    --imagenet-val "/mmfs1/data/yfcc-tmp/imagenet/val/" \
    --dataset-type webdataset \
    --precision amp \
    --gather-with-grad \
    --local-loss \
    --batch-size 512 \
    --workers 6 \
    --cosinereg 0.01 \
    --reg_threshold 0.3 \
    --report-to 'wandb' \
    --epochs 128 \
    --seed 0 \
    --zeroshot-frequency 2 \
    --wandb-project-name "greater_threshold" \
    --logs "/mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src/logs/greater_to_zero_threshold" \
    --name "ViT-B-16_b512_cosine0.01_thres0.3_lr5e-4_scratch" 
    