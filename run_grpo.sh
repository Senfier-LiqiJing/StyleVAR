#!/bin/bash
# StyleVAR GRPO (LoRA) training
# Single 48GB GPU, rank=64, G=8, bs=4
#
# Usage:
#   bash run_grpo.sh                    # default settings (nohup background)
#   bash run_grpo.sh --epochs 10        # override any arg
#   CUDA_VISIBLE_DEVICES=1 bash run_grpo.sh  # select GPU

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/grpo.log"
OUTPUT_DIR="${SCRIPT_DIR}/grpo_output"

mkdir -p "${OUTPUT_DIR}"

nohup python "${SCRIPT_DIR}/train_grpo.py" \
    --content_dir "${SCRIPT_DIR}/data/coco2017/images/train2017" \
    --style_dir "${SCRIPT_DIR}/data/wikiart" \
    --vae_ckpt "${SCRIPT_DIR}/ckpt/vae_ch160v4096z32.pth" \
    --sft_out_dir "${SCRIPT_DIR}/Output" \
    --out_dir "${OUTPUT_DIR}" \
    --lora_rank 64 --lora_alpha 128 \
    --G 8 --batch_size 4 \
    --lr 5e-5 \
    --epochs 5 \
    --save_every 200 \
    --exp_name grpo_lora_v1 \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

echo "GRPO training started in background (PID: $!)"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Monitor: tail -f ${LOG_FILE}"
