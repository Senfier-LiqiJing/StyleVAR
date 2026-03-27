#!/bin/bash
# StyleVAR curriculum mixed fine-tuning
# Single 48GB GPU, BF16, bs_per_gpu=12, effective_batch=48
#
# Usage:
#   bash run_train.sh              # default settings (nohup background)
#   bash run_train.sh --ep 5       # override any arg
#   CUDA_VISIBLE_DEVICES=1 bash run_train.sh  # select GPU

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/finetune.log"
OUTPUT_DIR="${SCRIPT_DIR}/Output"

mkdir -p "${OUTPUT_DIR}"

nohup torchrun --nproc_per_node=1 "${SCRIPT_DIR}/fine_tune.py" \
    --fp16 2 \
    --bs 48 --ac 4 \
    --tblr 5e-4 \
    --ep 10 \
    --data_path "${SCRIPT_DIR}/data/OmniStyle-150k" \
    --new_data_path "${SCRIPT_DIR}/data/ImagePulse" \
    --curriculum_start 0.3 --curriculum_end 0.7 \
    --clean_ckpt_path "${SCRIPT_DIR}/ckpt/style_var_d20_11_20_21.pth" \
    --exp_name curriculum_mixed_v1 \
    --save_every 200 \
    --wandb_run_id 0cy9tpl8 \
    --local_out_dir_path "${OUTPUT_DIR}" \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

echo "Training started in background (PID: $!)"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Monitor: tail -f ${LOG_FILE}"
