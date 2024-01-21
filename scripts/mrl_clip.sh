cd ~/lightning/open_clip/src
#torchrun --nproc_per_node 2 --master_port 3233 -m training.main \
lightning run model training/main.py \
    --model "ViT-B-16" \
    --train-data "/home/krmayank/data/medium/data/{00000000..00001919}.tar" \
    --imagenet-val="/home/krmayank/data/imagenet/imagenet_val/" \
    --dataset-type webdataset \
    --gather-with-grad \
    --local-loss \
    --force_mrl_loss \
    --mrl_loss_weights "1,1,1,1,1" \
    --mrl_dim_to_consider "768,384,192,96,48" \
    --batch-size 128 \
    --accum-freq 1 \
    --workers 4 \
    --epochs 2 \
    --warmup 5 \
    --zeroshot-frequency 1 \
    --save-frequency 1 \
    --seed 1234 \
    --train-num-samples 2048 \
#    --report-to 'wandb' \
#    --wandb-project-name "mrl_clip_training" \
#    --logs "/mmfs1/gscratch/krishna/mayank/clip_clone/open_clip/src/logs/mrl_clip" \
#    --name "mrl_clip_b512_accum_1_ep40_diffLogitScale_D082723_wl010150202503" 
