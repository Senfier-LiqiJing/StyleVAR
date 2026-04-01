#!/bin/bash
# StyleVAR SFT v2: content-preserving alpha schedule
# - Lower alphas at fine scales for better content detail preservation
# - Alpha jitter for robustness and inference-time style_strength control
# - Continue from best SFT checkpoint, NOT from scratch
#
# Changes from v1:
#   alpha_nums: 0.2..0.8 → 0.1..0.5 (halved at fine scales)
#   alpha_jitter: 0 → 0.15
#   bs: 48 → 64 (fill 48GB GPU)
#   ac: 4 → 4 (effective batch = 64*4 = 256)
#   Starts from pretrained VAR (NOT SFT v1, because alpha changed)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/finetune_v2.log"
OUTPUT_DIR="${SCRIPT_DIR}/Output_v2"

mkdir -p "${OUTPUT_DIR}"

nohup torchrun --nproc_per_node=1 "${SCRIPT_DIR}/fine_tune.py" \
    --fp16 2 \
    --bs 56 --ac 4 \
    --tblr 5e-4 \
    --ep 10 \
    --data_path "${SCRIPT_DIR}/data/OmniStyle-150k" \
    --new_data_path "${SCRIPT_DIR}/data/ImagePulse" \
    --concat_datasets \
    --alpha_nums "0.1_0.2_0.25_0.3_0.3_0.35_0.4_0.4_0.5_0.5" \
    --alpha_jitter 0.15 \
    --clean_ckpt_path "${SCRIPT_DIR}/ckpt/var_d20.pth" \
    --exp_name sft_v2_content_preserving \
    --wandb_project StyleVAR \
    --save_every 1000 \
    --val_every 4000 \
    --local_out_dir_path "${OUTPUT_DIR}" \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

echo "SFT v2 training started in background (PID: $!)"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Monitor: tail -f ${LOG_FILE}"
