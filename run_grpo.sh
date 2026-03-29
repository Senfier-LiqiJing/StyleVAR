#!/bin/bash
# StyleVAR GRPO training (full-param, single 49GB GPU)
#
# Usage:
#   bash run_grpo.sh                         # full-param (default)
#   bash run_grpo.sh --use_lora              # LoRA mode
#   bash run_grpo.sh --epochs 10             # override any arg
#   CUDA_VISIBLE_DEVICES=1 bash run_grpo.sh  # select GPU

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/grpo.log"
OUTPUT_DIR="${SCRIPT_DIR}/grpo_output"

mkdir -p "${OUTPUT_DIR}"

nohup python -u "${SCRIPT_DIR}/train_grpo.py" \
    --content_dir "${SCRIPT_DIR}/data/coco2017/images/train2017" \
    --style_dir "${SCRIPT_DIR}/data/wikiart" \
    --vae_ckpt "${SCRIPT_DIR}/ckpt/vae_ch160v4096z32.pth" \
    --sft_out_dir "${SCRIPT_DIR}/Output" \
    --out_dir "${OUTPUT_DIR}" \
    --G 4 --batch_size 12 \
    --lr 2e-6 \
    --epochs 5 \
    --save_every 200 \
    --exp_name grpo_fullparam_v1 \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

echo "GRPO training started in background (PID: $!)"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Monitor: tail -f ${LOG_FILE}"
