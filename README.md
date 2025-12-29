# StyleVAR: Controllable Image Style Transfer via Visual Autoregressive Modeling

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://github.com/Senfier-LiqiJing/StyleVAR)

**StyleVAR** is a reference-based image style transfer framework built upon Visual Autoregressive Modeling (VAR). It addresses the challenge of balancing content preservation and style intensity by formulating style transfer as a conditional discrete sequence modeling task. By introducing a novel **Blended Cross-Attention** mechanism, StyleVAR effectively injects artistic textures while maintaining the semantic structure of the content image.

> **Note**: This is a course/research project final report implementation.

---

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Training](#-training)
- [Inference](#-inference)
- [Results](#-results)
- [Team](#-team)
- [Citation](#-citation)

---

## 📝 Introduction

Traditional style transfer methods (CNN or Diffusion-based) often struggle to balance computational efficiency with generation quality or require complex prompt engineering. **StyleVAR** formulates the style transfer task as **conditional discrete sequence modeling** in a multi-scale latent space.

**Key Features:**
- **VAR Framework**: Leverages the "Next-Scale Prediction" paradigm for efficient autoregressive image generation.
- **Blended Cross-Attention**: A novel attention mechanism that uses style and content features as Queries to guide the autoregressive generation of the target image, ensuring style injection without breaking VAR's continuity.
- **Multi-Scale Generation**: Progressively refines images from coarse to fine scales, ensuring structural consistency.

---

## 🏗️ Methodology

### Framework

StyleVAR utilizes a VQ-VAE to tokenize images into discrete codes and an autoregressive transformer to predict target tokens scale-by-scale. We introduce a **Blended Cross-Attention** mechanism within each transformer block.

The feature update rule is defined as:

$$h_{new} = h + [\alpha \cdot \text{Attn}(Q=s^k, K=h, V=h) + (1-\alpha) \cdot \text{Attn}(Q=e^k, K=h, V=h)]$$

Where:
- $h$: Current target image history features (acting as Key and Value).
- $s^k, e^k$: Style and Content features at scale $k$ (acting as Queries).
- $\alpha$: Blended coefficient controlling the style-content balance.

![StyleVAR Framework](assets/figure1_framework.png)
*Figure 1: The framework of the proposed StyleVAR. The Blended Cross-Attention mechanism injects style and content information into the autoregressive generation process.*

---

## ⚙️ Installation

Please ensure your environment meets the following requirements (based on VAR dependencies):

- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU (A100 recommended for training as per report configuration)

```bash
git clone [https://github.com/Senfier-LiqiJing/StyleVAR.git](https://github.com/Senfier-LiqiJing/StyleVAR.git)
cd StyleVAR
pip install -r requirements.txt
