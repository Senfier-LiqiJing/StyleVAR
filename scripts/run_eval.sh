#!/bin/bash
# One-click evaluation: runs eval_grpo.py on 3 datasets with sane defaults.
#
# Usage:
#   bash run_eval.sh                                    # default: eval ckpt/grpo-best.pth
#   bash run_eval.sh ckpt/sft-best.pth                  # eval SFT baseline
#   bash run_eval.sh ckpt/grpo-best.pth my_eval         # custom out_dir
#   CKPT=ckpt/x.pth OUT=y bash run_eval.sh              # env-var form
#   bash run_eval.sh --lpips_backbone alex              # extra args passed through

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Positional: $1=ckpt path, $2=out_dir; fall back to env var or defaults
CKPT="${1:-${CKPT:-${ROOT}/ckpt/grpo-best.pth}}"
OUT="${2:-${OUT:-${ROOT}/eval_out}}"

# If $1 starts with '--', it's a flag, not a ckpt path — restore defaults
if [[ "${CKPT}" == --* ]]; then
    shift 0  # keep positional args
    CKPT="${ROOT}/ckpt/grpo-best.pth"
else
    shift $(( $# > 0 ? 1 : 0 ))  # consume $1
    if [[ -n "${1:-}" && "${1}" != --* ]]; then shift 1; fi  # consume $2 if present
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUT}"

echo "=========================================="
echo "  StyleVAR Evaluation"
echo "=========================================="
echo "  CKPT:       ${CKPT}"
echo "  OUT_DIR:    ${OUT}"
echo "  Datasets:   OmniStyle (50) + ImagePulse (50) + COCO+WikiArt (50)"
echo "=========================================="

python -u "${ROOT}/eval/eval_grpo.py" \
    --ckpt          "${CKPT}" \
    --vae_ckpt      "${ROOT}/ckpt/vae_ch160v4096z32.pth" \
    --clip_local_dir "${ROOT}/ckpt/clip-vit-base-patch32" \
    --omnistyle_root  "${ROOT}/data/OmniStyle-150k" \
    --imagepulse_root "${ROOT}/data/ImagePulse" \
    --coco_root       "${ROOT}/data/coco2017/images/train2017" \
    --wikiart_root    "${ROOT}/data/wikiart" \
    --omnistyle_n 50 --imagepulse_n 50 --cocowiki_n 50 \
    --top_k 900 --top_p 0.96 \
    --seed 42 \
    --out_dir "${OUT}" \
    "$@" \
    2>&1 | tee "${OUT}/eval.log"

echo ""
echo "[eval] results saved in ${OUT}/:"
echo "  - summary.csv         (machine readable metrics)"
echo "  - summary.json        (same + metadata)"
echo "  - eval.log            (full run log)"
echo "  - samples/<dataset>/  (first 8 comparison grids per dataset)"
