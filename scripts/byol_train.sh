cd /mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src
torchrun --nproc_per_node 2 --master_port 1235 -m training.main \
    --model "ViT-B-16" \
    --train-data "/mmfs1/data/yfcc-tmp/cc_3m/train_shards/shard_{000000..003318}.tar" \
    --imagenet-val "/mmfs1/data/yfcc-tmp/imagenet/val/" \
    --dataset-type webdataset \
    --precision amp \
    --gather-with-grad \
    --local-loss \
    --force_byol_clip \
    --batch-size 8 \
    --workers 6 \
    --epochs 128 \
    --seed 0 \
    --zeroshot-frequency 2 \
    --logs "/mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src/logs/byol_clip" \
    --name "running_debugging_byol"
