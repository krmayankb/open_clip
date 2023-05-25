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
    --lr 0.00003 \
    --epochs 42 \
    --cosinereg 0.001 \
    --reg_threshold 0.3 \
    --zeroshot-frequency 1 \
    --freeze_text \
    --workers 6 \
    --seed 0 \
    --report-to 'wandb' \
    --wandb-project-name "freezed_finetuning" \
    --resume /mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src/logs/scratch/2023_04_07-10_29_43-model_ViT-B-16-lr_0.0005-b_512-j_6-p_amp-reg_0.0/checkpoints/epoch_32.pt\
    --logs "/mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src/logs/text_freezed_finetuning" \
    --name "ViT-B-16_cos0.001_thres0.3_lr5e-4_textfreezed" 
    