cd /mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src
torchrun --nproc_per_node 2 -m training.main \
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
    --report-to 'wandb' \
    --epochs 128 \
    --wandb-project-name "threshold_finetuning" \
    --seed 0 \
    