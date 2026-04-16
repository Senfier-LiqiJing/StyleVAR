#!/bin/bash
# StyleVAR GRPO v3: DreamSim + PANW + G=16
#
# Key features:
#   - Paired data (OmniStyle + ImagePulse) with DreamSim reward
#   - PANW (Per-Action Normalization Weighting, alpha=0.7): upweights coarse scales
#   - G=16 rollouts for better advantage estimation (serial, no extra memory)
#   - KL coef=0.1 (matched to AR-GRPO paper)
#   - Only 400 steps (short alignment, not deep RL)
#   - Starts from SFT v2 best checkpoint
#
# Usage:
#   bash run_grpo.sh                                    # fresh start from SFT
#   bash run_grpo.sh --var_ckpt path/to/sft.pth         # specify SFT checkpoint
#   bash run_grpo.sh --resume path/to/grpo_ckpt.pth     # resume GRPO training
#   CUDA_VISIBLE_DEVICES=1 bash run_grpo.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/grpo.log"
OUTPUT_DIR="${SCRIPT_DIR}/grpo_output_v3"

# HuggingFace mirror for mainland China
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p "${OUTPUT_DIR}"

nohup python -u "${SCRIPT_DIR}/train_grpo.py" \
    --paired_data \
    --old_data_path "${SCRIPT_DIR}/data/OmniStyle-150k" \
    --new_data_path "${SCRIPT_DIR}/data/ImagePulse" \
    --vae_ckpt "${SCRIPT_DIR}/ckpt/vae_ch160v4096z32.pth" \
    --sft_out_dir "${SCRIPT_DIR}/Output_v2" \
    --out_dir "${OUTPUT_DIR}" \
    --use_lora --lora_rank 256 --lora_alpha 512 \
    --G 16 --batch_size 16 \
    --lr 1e-5 --kl_coef 0.1 --kl_target 0 --grad_clip 1.0 \
    --panw_alpha 0.7 \
    --epochs 1 \
    --save_every 100 \
    --use_dreamsim --dreamsim_scale 5.0 \
    --exp_name grpo_v3_panw \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

echo "GRPO v3 (DreamSim + PANW + G=16) started in background (PID: $!)"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Monitor: tail -f ${LOG_FILE}"
