#!/bin/bash
# StyleVAR curriculum mixed fine-tuning
# Single 48GB GPU, BF16, bs_per_gpu=12, effective_batch=48
#
# Usage:
#   bash run_train.sh              # default settings
#   bash run_train.sh --ep 5       # override any arg
#   CUDA_VISIBLE_DEVICES=1 bash run_train.sh  # select GPU

set -e

torchrun --nproc_per_node=1 fine_tune.py \
    --fp16 2 \
    --bs 48 --ac 4 \
    --tblr 5e-4 \
    --ep 10 \
    --data_path ./data/OmniStyle-150k \
    --new_data_path ./data/ImagePulse \
    --curriculum_start 0.3 --curriculum_end 0.7 \
    --clean_ckpt_path ./ckpt/style_var_d20_clean_fp32.pth \
    --vanilla_ckpt_path ./ckpt/var_d20.pth \
    --exp_name curriculum_mixed_v1 \
    --save_every 200 \
    "$@"
