#!/bin/bash
# StyleVAR GRPO v5: DreamSim reward (v3-proven) + Iterative merge + PANW
#
# Rationale:
#   - v4 (CLIP reward) regressed: reward_ema peaked 4.216 at step 55 then fell to 4.205 at step 135.
#     CLIP's narrow cosine range (~0.08) gives noisy advantages -> weak gradient signal.
#   - v3 (DreamSim reward, fixed ref) climbed smoothly -3.6 -> -3.35 over 800 steps.
#   - v5 = v3's proven DreamSim reward + v4's peak-triggered iterative merge.
#
# Iterative merge safety rails (unchanged from v4):
#   - merge_cooldown=300:    min 300 steps between merges
#   - merge_min_gain=0.05:   only merge if reward_ema improved by 0.05
#   - merge_patience=50:     must hold peak for 50 steps
#   - save_peak_lora:        merge uses peak snapshot, not current possibly-degraded weights
#   - merge_kl_threshold=2.0: emergency merge if KL runs away
#
# Usage:
#   bash run_grpo_v5.sh                                    # starts from SFT (Output_v2)
#   bash run_grpo_v5.sh --var_ckpt ckpt/sft_plus_v3.pth    # continue from v3-merged base
#   bash run_grpo_v5.sh --resume path/to/grpo_ckpt.pth     # resume v5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/grpo_v5.log"
OUTPUT_DIR="${SCRIPT_DIR}/grpo_output_v5"

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
    --use_dreamsim --dreamsim_scale 5.0 \
    --merge_cooldown 300 \
    --merge_min_gain 0.05 \
    --merge_patience 50 \
    --merge_kl_threshold 2.0 \
    --save_peak_lora \
    --exp_name grpo_v5_dreamsim_iter \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "GRPO v5 (DreamSim + iterative merge + PANW) started (PID: ${PID})"
echo "Log:    ${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Monitor: tail -f ${LOG_FILE}"
echo "Stop:    kill ${PID}"
