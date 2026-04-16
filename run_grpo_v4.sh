#!/bin/bash
# StyleVAR GRPO v4: CLIP reward + Iterative merge + PANW
#
# Changes vs v3:
#   - Reward: CLIP(gen, target) cosine similarity (pilot 2: rho=0.79 vs DreamSim's 0.64)
#   - Iterative merge: peak-triggered (safer than fixed interval — past failures
#     happened with fixed interval before PANW stabilized training)
#   - Keep: G=16, PANW alpha=0.7, kl_coef=0.1, paired data
#
# Iterative-merge safety rails (addresses past failure modes):
#   - merge_cooldown=300:  no merges in first 300 steps, then min 300 between
#   - merge_min_gain=0.05: only merge if reward_ema improved by this much
#   - merge_patience=50:   must sit at peak for 50 steps before confirming
#   - save_peak_lora:      merge uses the peak snapshot, not current degraded weights
#   - merge_kl_threshold=2.0: emergency merge if KL runs away
#
# Usage:
#   bash run_grpo_v4.sh                                    # fresh from SFT
#   bash run_grpo_v4.sh --var_ckpt path/to/sft.pth         # custom SFT ckpt
#   bash run_grpo_v4.sh --resume path/to/grpo_ckpt.pth     # resume GRPO training

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/grpo_v4.log"
OUTPUT_DIR="${SCRIPT_DIR}/grpo_output_v4"
CLIP_DIR="${CLIP_DIR:-${SCRIPT_DIR}/ckpt/clip-vit-base-patch32}"

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

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
    --epochs 3 \
    --save_every 100 \
    --use_clip_reward --clip_scale 5.0 \
    --clip_local_dir "${CLIP_DIR}" \
    --merge_cooldown 300 \
    --merge_min_gain 0.05 \
    --merge_patience 50 \
    --merge_kl_threshold 2.0 \
    --save_peak_lora \
    --exp_name grpo_v4_clip_iter \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "GRPO v4 (CLIP + iterative merge + PANW) started (PID: ${PID})"
echo "Log:    ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Monitor: tail -f ${LOG_FILE}"
echo "Stop:    kill ${PID}"
