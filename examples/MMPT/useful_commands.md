```
CUDA_VISIBLE_DEVICES=5 python scripts/video_feature_extractor/extract.py --vdir /fb-agios-acai-efs/coin/videos --fdir /fb-agios-acai-efs/crosstask/crosstask_release/s3dg_features_vidoeclip --type=s3d --num_decoding_thread=20 --batch_size 32 --half_precision 1
```

```
CUDA_VISIBLE_DEVICES=3,4 python locallaunch.py projects/retri/videoclip/coin_videoclip_new_split.yaml --jobtype local_small
```

```
CUDA_VISIBLE_DEVICES=2,3 fairseq-train projects/retri/videoclip/crosstask_videoclip.yaml --user-dir mmpt --task mmtask --arch mmarch --criterion mmloss --distributed-world-size 2 --tensorboard-logdir runs/retri/videoclip/crosstask/run_0 --log-interval 1000 --fp16 --num-workers 4 --batch-size 1 --lr 5e-05 --clip-norm 2.0 --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler polynomial_decay --total-num-update 1000000 --warmup-updates 122 --weight-decay 0.0 --ddp-backend no_c10d --max-epoch 5 --restore-file runs/retri/videoclip/checkpoint_best.pt --reset-optimizer --reset-dataloader --reset-meters --save-dir runs/retri/videoclip/crosstask
```