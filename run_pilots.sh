#!/bin/bash
# Run both pilot studies sequentially with real-time output.
# Results go to pilot_results/ (heatmaps, CSVs, scatter plots).
#
# Usage:
#   bash run_pilots.sh                    # run both
#   bash run_pilots.sh p1                 # only pilot 1
#   bash run_pilots.sh p2                 # only pilot 2

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# ---- Paths (edit if different on your machine) ----
SFT_CKPT="Output_v2/ar-ckpt-best.pth"
GRPO_CKPT="/home/linux/StyleVAR/grpo_output_v3/grpo_best.pth"
VAE_CKPT="ckpt/vae_ch160v4096z32.pth"
CLIP_DIR="/home/linux/StyleVAR/ckpt/clip-vit-base-patch32"

CONTENT_DIR="data/coco2017/images/train2017"
STYLE_DIR="data/wikiart"
OLD_PAIRED="data/OmniStyle-150k"
NEW_PAIRED="data/ImagePulse"

OUT_DIR="pilot_results"
mkdir -p "${OUT_DIR}"

# Unbuffered python + real-time tee
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

WHICH="${1:-both}"

# ============ Pilot 1: scale-reward gradient heatmap ============
if [[ "${WHICH}" == "both" || "${WHICH}" == "p1" ]]; then
    echo ""
    echo "==================================================="
    echo "  Pilot 1: Scale × Reward Gradient Sensitivity"
    echo "==================================================="
    python -u pilot_scale_reward.py \
        --sft_ckpt     "${SFT_CKPT}" \
        --grpo_ckpt    "${GRPO_CKPT}" \
        --vae_ckpt     "${VAE_CKPT}" \
        --content_dir  "${CONTENT_DIR}" \
        --style_dir    "${STYLE_DIR}" \
        --clip_local_dir "${CLIP_DIR}" \
        --batch_size 2 --num_batches 8 \
        --out_dir      "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/pilot1.log"
    echo "[Pilot 1] done -> ${OUT_DIR}/scale_reward_heatmap.png"
fi

# ============ Pilot 2: DreamSim content-vs-GT correlation ============
if [[ "${WHICH}" == "both" || "${WHICH}" == "p2" ]]; then
    echo ""
    echo "==================================================="
    echo "  Pilot 2: DreamSim Content vs GT Correlation"
    echo "==================================================="
    python -u pilot_dreamsim_proxy.py \
        --old_data_path "${OLD_PAIRED}" \
        --new_data_path "${NEW_PAIRED}" \
        --clip_local_dir "${CLIP_DIR}" \
        --n_triplets 200 \
        --also_composite \
        --out_dir       "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/pilot2.log"
    echo "[Pilot 2] done -> ${OUT_DIR}/dreamsim_proxy_spearman.png"
fi

echo ""
echo "All done. Results in ${OUT_DIR}/"
ls -la "${OUT_DIR}/"
