#!/bin/bash
# Fix script: run this after setup_env.sh fails at flash-attn
# Usage: bash setup_env_fix.sh

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate var

echo "=== Installing flash-attn build deps ==="
pip install psutil packaging setuptools wheel

echo "=== Installing flash-attn ==="
pip install flash-attn==2.6.3 --no-build-isolation

echo "=== Installing remaining dependencies ==="
pip install \
    einops==0.8.2 \
    lpips==0.1.4 \
    numpy \
    pillow \
    scipy \
    tensorboard \
    tqdm \
    transformers \
    huggingface_hub \
    safetensors \
    pyyaml \
    typed-argument-parser

echo ""
echo "=== Fix complete! ==="
