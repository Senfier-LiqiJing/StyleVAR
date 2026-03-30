#!/bin/bash
# StyleVAR GRPO training (LoRA, fp32, single 49GB GPU)
#
# Usage:
#   bash run_grpo.sh                         # default (LoRA)
#   bash run_grpo.sh --epochs 20             # override any arg
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
    --use_lora --lora_rank 256 --lora_alpha 512 \
    --G 4 --batch_size 12 \
    --lr 5e-5 --kl_coef 0.01 --grad_clip 1.0 \
    --epochs 10 \
    --save_every 200 \
    --lam_content 1.0 --lam_style 1.0 --lam_ssim 1.0 --lam_tv 0.0 \
    --exp_name grpo_lora_v1 \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

echo "GRPO training started in background (PID: $!)"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Monitor: tail -f ${LOG_FILE}"
