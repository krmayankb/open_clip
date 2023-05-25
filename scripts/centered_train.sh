cd /mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src
torchrun --nproc_per_node 2 --master_port 1234 -m training.main \
    --model "ViT-B-16" \
    --train-data "/mmfs1/data/yfcc-tmp/cc_3m/train_shards/shard_{000000..003318}.tar" \
    --imagenet-val "/mmfs1/data/yfcc-tmp/imagenet/val/" \
    --dataset-type webdataset \
    --precision amp \
    --gather-with-grad \
    --local-loss \
    --batch-size 512 \
    --centered_clip \
    --workers 6 \
    --report-to 'wandb' \
    --epochs 128 \
    --seed 0 \
    --zeroshot-frequency 2 \
    --wandb-project-name "centered_clip" \
    --logs "/mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src/logs/centered_clip" \
    --name "ViT-B-16_b512_CenteredClip_lr5e-4_scratch_2" 