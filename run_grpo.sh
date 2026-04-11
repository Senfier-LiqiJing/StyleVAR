#!/bin/bash
# StyleVAR GRPO v2: DreamSim reward + paired data (OmniStyle + ImagePulse)
#
# Key changes from v1:
#   - Paired data: (content, style, target) triplets from SFT datasets
#   - Reward: single DreamSim(gen, target) — no LPIPS/Gram/SSIM mix
#   - G=8 rollouts (was 4) for better advantage signal
#   - batch_size=6 (was 12) to fit G=8 + DreamSim in memory
#   - Starts from SFT v2 best checkpoint
#
# Usage:
#   bash run_grpo.sh                 # default (LoRA)
#   CUDA_VISIBLE_DEVICES=1 bash run_grpo.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/grpo.log"
OUTPUT_DIR="${SCRIPT_DIR}/grpo_output_v2"

mkdir -p "${OUTPUT_DIR}"

nohup python -u "${SCRIPT_DIR}/train_grpo.py" \
    --paired_data \
    --old_data_path "${SCRIPT_DIR}/data/OmniStyle-150k" \
    --new_data_path "${SCRIPT_DIR}/data/ImagePulse" \
    --vae_ckpt "${SCRIPT_DIR}/ckpt/vae_ch160v4096z32.pth" \
    --sft_out_dir "${SCRIPT_DIR}/Output_v2" \
    --out_dir "${OUTPUT_DIR}" \
    --use_lora --lora_rank 256 --lora_alpha 512 \
    --G 8 --batch_size 16 \
    --lr 1e-5 --kl_coef 0.02 --kl_target 0 --grad_clip 1.0 \
    --merge_cooldown 200 --merge_min_gain 0.05 --merge_patience 100 \
    --epochs 10 \
    --save_every 200 \
    --use_dreamsim --dreamsim_scale 5.0 \
    --exp_name grpo_dreamsim_v2 \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

echo "GRPO v2 (DreamSim + paired) training started in background (PID: $!)"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Monitor: tail -f ${LOG_FILE}"
