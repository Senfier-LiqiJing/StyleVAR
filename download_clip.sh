#!/bin/bash
# Download openai/clip-vit-base-patch32 via HF mirror.
# Run this on any machine with internet; the resulting ckpt/clip-vit-base-patch32
# directory can be rsync'd / scp'd to the training machine.
#
# Usage:
#   bash download_clip.sh
#   bash download_clip.sh /custom/output/dir

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/ckpt/clip-vit-base-patch32}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"  # off by default

echo "[download_clip] OUT_DIR=${OUT_DIR}"

mkdir -p "${OUT_DIR}"

# Prefer huggingface-cli (comes with huggingface_hub>=0.14); fall back to Python.
if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download openai/clip-vit-base-patch32 \
        --local-dir "${OUT_DIR}" \
        --local-dir-use-symlinks False
else
    echo "[download_clip] huggingface-cli not found, using Python snapshot_download"
    python - <<PY
import os
from huggingface_hub import snapshot_download
out = snapshot_download(
    repo_id="openai/clip-vit-base-patch32",
    local_dir="${OUT_DIR}",
    local_dir_use_symlinks=False,
)
print(f"downloaded to: {out}")
PY
fi

echo ""
echo "[download_clip] Done."
echo "Transfer to training machine:"
echo "  rsync -avz ${OUT_DIR}/ user@gpu-box:${OUT_DIR}/"
echo ""
echo "Then run pilot with:"
echo "  python pilot_scale_reward.py ... --clip_local_dir ${OUT_DIR}"
