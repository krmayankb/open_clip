cd /mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src
torchrun --nproc_per_node 2 -m training.main \
    --model "ViT-B-16" \
    --train-data "/mmfs1/data/yfcc-tmp/cc_3m/train_shards/shard_{000000..003318}.tar" \
    --imagenet-val "/mmfs1/data/yfcc-tmp/imagenet/val/" \
    --dataset-type webdataset \
    --precision amp \
    --gather-with-grad \
    --batch-size 512 \
    --cosinereg 0.001 \
    --report-to 'wandb' \
    --resume /mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src/logs/2023_03_08-18_33_53-model_ViT-B-16-lr_0.0005-b_256-j_6-p_amp-reg_0.001/checkpoints/epoch_26.pt

