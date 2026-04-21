#!/bin/bash
# Download the AdaIN decoder weights (for the baseline in eval_grpo.py --also_adain).
#
# Source: naoto0804/pytorch-AdaIN on GitHub (the canonical PyTorch reimplementation
# of Huang & Belongie 2017). Decoder was trained by that author; MIT licensed.
#
# Usage:
#   bash download_adain.sh                       # default: ckpt/adain_decoder.pth
#   bash download_adain.sh /custom/path.pth      # custom output path

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-${SCRIPT_DIR}/ckpt/adain_decoder.pth}"

mkdir -p "$(dirname "${OUT}")"

# Official Google Drive URL (from naoto0804/pytorch-AdaIN README).
# We use gdown for reliable Google Drive download.
URL="https://drive.google.com/uc?id=1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr"

echo "[download_adain] target: ${OUT}"
echo "[download_adain] url:    ${URL}"

if command -v gdown >/dev/null 2>&1; then
    gdown -O "${OUT}" "${URL}"
else
    echo "[download_adain] gdown not found, installing..."
    pip install -q gdown
    gdown -O "${OUT}" "${URL}"
fi

if [[ -f "${OUT}" ]]; then
    size=$(du -h "${OUT}" | cut -f1)
    echo "[download_adain] done. ${OUT} (${size})"
    echo ""
    echo "Next: evaluate with AdaIN baseline:"
    echo "  bash run_eval.sh ckpt/grpo-best.pth eval_out --also_adain"
else
    echo "[download_adain] FAILED. Manually download from:"
    echo "  https://github.com/naoto0804/pytorch-AdaIN"
    echo "And place decoder.pth at: ${OUT}"
    exit 1
fi
