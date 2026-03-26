#!/bin/bash
# StyleVAR environment setup
# Usage: bash setup_env.sh

set -e

ENV_NAME="var"
PYTHON_VERSION="3.10"

echo "=== Creating conda env: ${ENV_NAME} (Python ${PYTHON_VERSION}) ==="
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

echo "=== Activating ${ENV_NAME} ==="
eval "$(conda init bash 2>/dev/null)"
conda activate ${ENV_NAME}

echo "=== Installing CUDA toolkit 12.1 ==="
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y

echo "=== Installing PyTorch 2.4.0 + CUDA 12.1 ==="
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing flash-attn (requires CUDA build) ==="
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
echo "=== Done! Activate with: conda activate ${ENV_NAME} ==="
