# StyleVAR: Controllable Image Style Transfer via Visual Autoregressive Modeling

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://github.com/Senfier-LiqiJing/StyleVAR)

**StyleVAR** is a reference-based image style transfer framework built upon Visual Autoregressive Modeling (VAR). This work formulates style transfer as conditional discrete sequence modeling in a multi-scale latent space, introducing a novel **Blended Cross-Attention** mechanism to balance content preservation and style intensity. The method preserves structural semantics of the content image while adopting the artistic texture of the style image through a principled attention-based conditioning strategy.

> **Authors**: Liqi Jing, Dingming Zhang, Peinian Li
> **Affiliation**: Duke University
> **Date**: December 17, 2025

---

## 📖 Table of Contents

- [Abstract](#-abstract)
- [Introduction](#-introduction)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Training](#-training)
- [Inference](#-inference)
- [Experimental Results](#-experimental-results)
- [Limitations and Discussion](#-limitations-and-discussion)
- [Future Work](#-future-work)
- [Project Roadmap](#-project-roadmap)
- [Team Contributions](#-team-contributions)
- [References](#-references)

---

## 📄 Abstract

This project studies reference-based image style transfer: given a content image and a style image, the goal is to generate an output that preserves the structural semantics of the content while adopting the artistic texture of the style. We build on the Visual Autoregressive Modeling (VAR) framework and formulate style transfer as conditional discrete sequence modeling in a learned latent space. Images are decomposed into multi-scale representations and tokenized into discrete codes by a VQ-VAE; a transformer then autoregressively models the distribution of target tokens conditioned on style and content tokens.

To inject style and content information, we introduce a **blended cross-attention mechanism** in which the evolving target representation attends to its own history, while style and content features act as queries that decide which aspects of this history to emphasize. A scale-dependent blending coefficient controls the relative influence of style and content at each stage, encouraging the synthesized representation to align with both the content structure and the style texture without breaking the autoregressive continuity of VAR.

We fine-tune this StyleVAR model from a pretrained VAR checkpoint on a large triplet dataset of content–style–target images and evaluate it on held-out pairs. Qualitative results indicate that the method can transfer texture while maintaining semantic structure, especially for landscapes and architectural scenes, while a generalization gap on internet images and difficulty with human faces highlight the need for better content diversity and stronger structural priors.

---

## 📝 Introduction

Reference-based image style transfer aims to generate an image that preserves the spatial layout and object semantics of a content image while adopting the colors, textures, and local patterns of a style image. This setting is valuable in artistic creation, visual prototyping, and controllable data augmentation, where users wish to restyle an existing scene without altering its high-level semantic meaning.

### Motivation and Challenges

Balancing content preservation and style strength presents fundamental challenges:
- **Content-Style Trade-off**: Overemphasis on content results in weak stylization; overemphasis on style may distort object shapes or introduce artifacts that break semantic coherence.
- **Style Diversity**: Styles vary widely—some primarily change global tone, while others rely on fine-grained textures and patterns.
- **Computational Efficiency**: Diffusion-based approaches require many iterative denoising steps, leading to slow sampling and high computational cost.

### Our Approach

StyleVAR adopts the Visual Autoregressive Modeling (VAR) framework and casts style transfer as **conditional discrete sequence modeling** in a multi-scale latent space. Each image is decomposed into a hierarchy of feature maps and tokenized into discrete codes by a VQ-VAE encoder. The target image is generated scale by scale, with each set of target tokens conditioned on:
1. The history of previously generated tokens
2. Corresponding style and content tokens at each scale

**Key Innovation**: The **Blended Cross-Attention** mechanism allows style and content features (as Queries) to selectively emphasize relevant aspects of the target's autoregressive history (Keys and Values), ensuring structural continuity while enabling effective style transfer.

---

## 🏗️ Methodology

### 2.1 Blended Cross-Attention Autoregressive Modeling

#### Formulation

In the context of style transfer, the objective is to predict a target image that preserves the structural semantics of a content image $x_c$ while adopting the artistic texture of a style image $x_s$. Adopting the framework of Visual Autoregressive Modeling (VAR), we decompose images into multi-scale representations. Each scale's feature map is tokenized into discrete tokens.

Formally, the style image tokens are denoted as $S = \{s^1, s^2, \ldots, s^K\} = \mathcal{E}(x_s)$, and the content image tokens as $C = \{c^1, c^2, \ldots, c^K\} = \mathcal{E}(x_c)$, where $K$ represents the total number of scales and $\mathcal{E}(\cdot)$ denotes the VQ-VAE tokenization process.

The generation of the target image, denoted as $R = \{r^1, r^2, \ldots, r^K\}$, proceeds in a scale-wise autoregressive manner:

$$P(x|x_s, x_c) = \prod_{k=1}^{K} P(r^k | r^{<k}, s^k, c^k)$$

where $r^k$ denotes the target features at the $k$-th scale, and $r^{<k} = r^{1:k-1}$ represents the history of generated target features prior to the $k$-th scale.

#### Model Structure

Within each transformer block, the feature update process is expressed as:

$$h_{new} = h + [\alpha \cdot \text{Attn}(Q=s^k, K=h, V=h) + (1-\alpha) \cdot \text{Attn}(Q=c^k, K=h, V=h)]$$

Where:
- $h$: Input target features at stage $k$ (or output of the preceding transformer block)
- $s^k, c^k$: Style and Content features at scale $k$ (acting as Queries)
- $\alpha_k$: Heuristic hyperparameter governing the blending ratio between style and content information
- Keys ($K$) and Values ($V$): Target feature history

![StyleVAR Framework](assets/figure1_framework.png)
*Figure 1: The framework of the proposed StyleVAR. The Blended Cross-Attention mechanism injects style and content information into the autoregressive generation process.*

### 2.2 Training and Inference

#### Inference Process

The inference process begins by initializing the start token at the first scale using content features extracted via a ResNet-18 backbone, projected to the embedding dimension via an MLP. A critical component is the progressive accumulation of features:

1. **Pre-calculation**: Style and content features are fully observable; their multi-scale ground truth tokens are pre-calculated via VQ-VAE decomposition
2. **Cumulative Generation**: A cumulative feature map $\hat{f}$ is maintained; at each step, generated tokens $r^k$ are quantized and added to $\hat{f}$ in a residual manner
3. **Downsampling**: $\hat{f}$ is downsampled to the appropriate resolution to serve as input for the next scale

#### Training Strategy

During training, we employ a teacher-forcing strategy:
1. Concatenate the start token with ground truth tokens of the target image across all scales
2. Following the vanilla VAR paradigm, predict logits for stages 1 to $K$ in parallel
3. Calculate Cross-Entropy loss between predicted logits and ground truth codebook indices
4. Input at any stage $k$ is the accumulation of ground truth features from all preceding scales (1 to $k-1$)

### 2.3 Design Rationale: Attention Configuration

#### Preserving Autoregressive Continuity

A critical design decision in StyleVAR is assigning target image features to the Key ($K$) and Value ($V$) roles, while assigning style/content features to the Query ($Q$) role. This configuration diverges from standard cross-attention mechanisms but is essential for preserving the "next-scale prediction" paradigm of VAR.

By designating the target feature history as $K$ and $V$, the attention mechanism explicitly aggregates information from the target's own past. The style and content features (as $Q$) act as a "search query", determining which parts of the target's history are most relevant to emphasize for the current generation step.

#### Theoretical Viability

By setting $V$ as the target features, the output of the attention block becomes a linear combination of the target's own history. While this does not directly "copy" pixels from the style image, the style-guided re-weighting (via the $Q \times K^T$ score) is theoretically sufficient to modulate the generative trajectory, effectively steering the autoregressive process to adopt the desired stylistic characteristics while maintaining structural integrity.

---

## ⚙️ Installation

Please ensure your environment meets the following requirements (based on VAR dependencies):

- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU (A100 recommended for training as per report configuration)

```bash
git clone https://github.com/Senfier-LiqiJing/StyleVAR.git
cd StyleVAR
pip install -r requirements.txt
```

---

## 📊 Dataset

### OmniStyle-150K Dataset

We utilized the **OmniStyle-150K** dataset for training, which consists of 143,992 triplets: (content image, style image, target image). The data is structured such that each target image is paired with its corresponding source content and style inputs.

**Data Augmentation**: To enhance model robustness and force the network to learn fine-grained structural and textural details, we applied data augmentation during preprocessing:
- **Content Images**: Rotation and brightness adjustments
- **Style Images**: Random cropping

**Dataset Composition Analysis**: While the dataset contains approximately 150k triplets, further analysis revealed these are generated from a limited pool of approximately 1,800 unique content images. This data imbalance has implications for model generalization, as discussed in the [Limitations](#-limitations-and-discussion) section.

---

## 🚀 Training

### Training Configuration

**Initialization**: We initialized StyleVAR weights using the pre-trained vanilla VAR model checkpoint. The VQ-VAE component was frozen while fine-tuning the full 600M parameters of the transformer.

**Architecture Adaptation**: Since StyleVAR utilizes a dual-stream input (target features and content/style condition features), the original projection layers of vanilla VAR were duplicated to initialize distinct projection layers for both the target and condition streams. Feed-Forward Networks (FFN) were initialized with the original VAR parameters.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Total Epochs | 8 |
| Learning Rate (Epochs 1-6) | 5 × 10⁻⁴ |
| Learning Rate (Epochs 7-8) | 1 × 10⁻⁴ |
| Physical Batch Size per GPU | 4 |
| Gradient Accumulation Steps | 128 |
| Effective Global Batch Size | 1,024 |
| Hardware | 2× NVIDIA A100 (40GB) |

### Training Command

```bash
python train.py \
  --data_path /path/to/OmniStyle-150K \
  --batch_size 4 \
  --grad_accum 128 \
  --epochs 8 \
  --lr 5e-4 \
  --lr_decay_epoch 6
```

---

## 🎨 Inference

### Autoregressive Generation Process

The inference pipeline follows a stage-wise autoregressive generation strategy:

1. **Condition Preparation**: Pre-encode style and content images via VQ-VAE to obtain multi-scale token sequences $S$ and $C$
2. **Initialization**: Initialize the start token using content features extracted via ResNet-18 backbone
3. **Stage-wise Generation**: For each scale $k = 1$ to $K$:
   - Pass current input through transformer to obtain logits
   - Sample tokens $r^k$ using top-k/top-p sampling
   - Look up quantized features from codebook
   - Accumulate features to cumulative map $\hat{f}$
   - Downsample $\hat{f}$ for next scale input (if $k < K$)
4. **Image Reconstruction**: Decode cumulative feature map $\hat{f}$ using VQ-VAE decoder

### Inference Command

```bash
python inference.py \
  --content_image /path/to/content.jpg \
  --style_image /path/to/style.jpg \
  --output_path /path/to/output.jpg \
  --checkpoint /path/to/checkpoint.pth \
  --sampling_method top_p \
  --top_p 0.95
```

### Performance

- **Average Inference Time**: ~0.40 seconds per image (256×256 resolution)
- **Throughput**: ~2.48 FPS on NVIDIA A100 GPU

---

## 📈 Experimental Results

### 3.1 Quantitative Analysis

After 8 epochs of fine-tuning, the model demonstrated promising convergence:
- **Mean Accuracy** (averaged across all scales): 14.72% top-1 accuracy
- **Tail Accuracy** (final resolution scale): 16.26% top-1 accuracy

### 3.2 Benchmark Comparison: StyleVAR vs. AdaIN

We benchmarked StyleVAR on 500 randomly selected style-content image pairs:
- **Style Images**: Randomly selected from WikiArt dataset
- **Content Images**: Randomly selected from MS-COCO dataset

#### Quantitative Results

| Metric | StyleVAR | AdaIN | Better |
|--------|----------|-------|--------|
| **Speed (FPS)** | 2.48 | 317.97 | AdaIN ↑ |
| **Avg Inference Time (s)** | 0.4031 | 0.0031 | AdaIN ↑ |
| **Style Preservation** (Style Loss) ↓ | 0.1081 | **0.0983** | AdaIN |
| **Content Preservation** (Content Loss) ↓ | **119.94** | 177.23 | **StyleVAR** |
| **Structure Preservation** (SSIM) ↑ | **0.3224** | 0.1884 | **StyleVAR** |
| **Perceptual Distance** (LPIPS) ↓ | **0.6297** | 0.7712 | **StyleVAR** |

**Key Findings**:
- **Content & Structure**: StyleVAR achieves **significantly better content preservation** (119.94 vs. 177.23) and **structure preservation** (0.3224 vs. 0.1884 SSIM)
- **Perceptual Quality**: StyleVAR demonstrates **lower perceptual distance** (0.6297 vs. 0.7712 LPIPS), indicating more realistic outputs
- **Style Transfer**: AdaIN shows slightly better style loss (0.0983 vs. 0.1081), suggesting more aggressive style transfer
- **Efficiency Trade-off**: AdaIN is ~128× faster, reflecting the computational cost of autoregressive generation

### 3.3 Training Dynamics

![Training Loss and Accuracy](assets/figure2_training.png)
*Figure 2: Loss and accuracy of training set across iterations. The model demonstrates consistent convergence with both mean accuracy and tail accuracy improving throughout training.*

### 3.4 Qualitative Analysis

Generated samples demonstrate that StyleVAR successfully transfers artistic textures while maintaining the semantic structure of content images:

![Qualitative Results](assets/figure3_results.png)
*Figure 3: The generated images demonstrate that the model successfully transfers texture while maintaining the semantic structure of the content. (Left: Content Image, Middle: Style Image, Right: Generated Output)*

**Observations**:
- **Landscapes & Architecture**: Excellent texture transfer with preserved spatial layout
- **Complex Textures**: Successfully adopts fine-grained artistic patterns (e.g., brushstrokes, color palettes)
- **Semantic Coherence**: Maintains object boundaries and semantic structure

---

## ⚠️ Limitations and Discussion

Despite strong performance on training and validation sets, qualitative evaluation revealed several important limitations:

### Generalization Gap

**Observation**: Performance degradation when testing on unseen images collected from the internet, indicating overfitting to the training distribution.

**Root Cause Analysis**: The OmniStyle-150K dataset contains ~150k triplets generated from only ~1,800 unique content images. Given StyleVAR's capacity (600M parameters), the model likely memorized structural priors of this limited content set rather than learning generalized content representations.

### Domain-Specific Performance Disparity

**Strong Performance**: Landscapes and architectural scenes
**Weak Performance**: Human faces

**Hypothesized Causes**:
1. **Complexity**: Facial topology is significantly more complex and sensitive to structural deformation than natural scenes
2. **Perceptual Sensitivity**: Human visual perception is acutely sensitive to structural anomalies in facial features
3. **Training Data Distribution**: Limited representation of human faces in the training dataset

### Computational Cost

While StyleVAR achieves superior content preservation and perceptual quality compared to AdaIN, it is ~128× slower (2.48 FPS vs. 317.97 FPS), limiting real-time applications.

---

## 🔮 Future Work

We plan to address generalization, controllability, and training efficiency through the following research directions:

### 5.1 Data Augmentation and Diversification

**Objective**: Improve generalization to diverse content types

**Strategy**:
- Expand training data with more diverse content images, particularly in challenging semantic domains (human faces, complex objects)
- Explore regularization and augmentation strategies that encourage learning transferable structural patterns
- Balance dataset composition to prevent memorization of limited content priors

### 5.2 Classifier-Free Style Guidance

**Objective**: Enable user-controllable style strength at inference time

**Approach**: Extend StyleVAR with classifier-free guidance mechanism:
1. **Training**: Occasionally drop style conditioning to teach the model both style-conditioned and style-agnostic predictions
2. **Inference**: Interpolate between style-conditioned and unconditional predictions when sampling tokens
3. **User Control**: Continuous dial for trading off content fidelity against stylistic intensity

**Benefits**: Complements existing blended attention design by providing straightforward runtime control without retraining

### 5.3 GRPO-Driven Unsupervised Learning

**Objective**: Transcend limitations of supervised training with paired ground truth

**Methodology**: Implement second-stage unsupervised fine-tuning via **Group Relative Policy Optimization (GRPO)**:

#### GRPO Framework for Visual Generation

1. **Policy Network**: StyleVAR model serves as the policy
2. **Sampling**: For each content-style pair, sample a group of diverse outputs
3. **Reward Signal**: Evaluate outputs using perceptual metrics:
   - VGG-based style loss
   - VGG-based content loss
   - LPIPS perceptual distance
   - SSIM structural similarity
4. **Optimization**: Optimize policy to favor outputs with higher relative rewards within the group

#### Advantages Over Actor-Critic Methods

- **Memory Efficiency**: Eliminates need for separate Critic model
- **Scalability**: Allocate more resources to batch size and context length
- **Direct Optimization**: Minimize perceptual loss directly through non-differentiable discrete sampling

**Expected Impact**: Enable the model to learn perceptual objectives without relying on paired ground truth, potentially improving generalization and reducing dataset bias.

### 5.4 Adaptive Blending Strategies

**Objective**: Data-driven strategies for setting or adapting the blending coefficient $\alpha$

**Approaches**:
- Learn scale-dependent and content-dependent blending coefficients
- Automatic adjustment to different content-style pairs
- Reduce failure cases in underrepresented or structurally sensitive domains

---

## ✅ Project Roadmap

### Completed Tasks

#### Foundation & Architecture
- [x] Literature review on VAR, VQ-VAE, and style transfer methods
- [x] Design blended cross-attention mechanism
- [x] Implement StyleVAR transformer architecture
- [x] Adapt VAR framework for dual-stream conditioning (style + content)
- [x] Initialize from pretrained VAR checkpoint

#### Data & Training Pipeline
- [x] Prepare OmniStyle-150K dataset
- [x] Implement data augmentation (rotation, brightness, cropping)
- [x] Implement teacher-forcing training strategy
- [x] Configure distributed training on 2× A100 GPUs
- [x] Implement gradient accumulation for effective batch size 1024
- [x] Fine-tune for 8 epochs with learning rate schedule

#### Inference & Evaluation
- [x] Implement autoregressive inference pipeline
- [x] Implement progressive feature accumulation
- [x] Integrate top-k/top-p sampling
- [x] Benchmark against AdaIN baseline (N=500)
- [x] Evaluate on WikiArt (style) and MS-COCO (content)
- [x] Compute metrics: Style Loss, Content Loss, SSIM, LPIPS
- [x] Qualitative analysis on validation set
- [x] Generate visualization figures

#### Documentation & Reporting
- [x] Write final project report
- [x] Document methodology and attention design rationale
- [x] Analyze limitations and generalization gap
- [x] Prepare training dynamics plots
- [x] Create qualitative result figures
- [x] Update README with academic refinement

### Pending Tasks

#### Generalization Improvements
- [ ] Expand dataset with more diverse content images
- [ ] Augment training data with human face domain
- [ ] Implement advanced regularization strategies
- [ ] Test on diverse internet image benchmarks
- [ ] Conduct ablation study on data diversity

#### Controllability Enhancements
- [ ] Implement classifier-free style guidance
- [ ] Train with conditional dropout for style conditioning
- [ ] Add inference-time style strength control
- [ ] Implement adaptive blending coefficient learning
- [ ] User study on controllability effectiveness

#### Reinforcement Learning Integration (GRPO)
- [ ] Implement GRPO training framework for visual generation
- [ ] Design group sampling strategy for diverse outputs
- [ ] Implement perceptual reward functions (VGG, LPIPS, SSIM)
- [ ] Configure reward aggregation and normalization
- [ ] Second-stage unsupervised fine-tuning with GRPO
- [ ] Evaluate GRPO impact on perceptual quality
- [ ] Compare GRPO vs. supervised baseline
- [ ] Memory profiling and optimization for GRPO

#### Efficiency & Deployment
- [ ] Profile computational bottlenecks
- [ ] Explore model distillation for faster inference
- [ ] Implement mixed-precision inference
- [ ] Optimize memory usage during generation
- [ ] Deploy demo web interface
- [ ] Create user-friendly inference scripts

#### Extended Evaluation
- [ ] Benchmark on additional style transfer datasets
- [ ] Human evaluation study (perceptual quality, preference)
- [ ] Ablation study on blending coefficient $\alpha$
- [ ] Ablation study on number of scales $K$
- [ ] Compare against diffusion-based style transfer methods
- [ ] Analyze failure cases systematically

---

## 👥 Team Contributions

**Liqi Jing**: Implementing the blended attention module for StyleVAR and benchmark evaluation

**Dingming Zhang**: Implementing the training framework and LoRA fine-tuning pipeline

**Peinian Li**: Implementing the dataset loader and visualization tools

---

## 📚 References

[1] Tian, K., Jiang, Y., Yuan, Z., Peng, B., & Wang, L. (2024). Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*.

[2] Wang, Y., Liu, R., Lin, J., Liu, F., Yi, Z., Wang, Y., & Ma, R. (2025). OmniStyle: Filtering High Quality Style Transfer Data at Scale. In *2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[3] Zhang, Y., Huang, N., Tang, F., Huang, H., Ma, C., Dong, W., & Xu, C. (2023). Inversion-Based Style Transfer with Diffusion Models. In *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 10077–10086).

[4] Huang, X., & Belongie, S. (2017). Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization. In *ICCV 2017*.

[5] Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... Zitnick, C. L. (2014). Microsoft COCO: Common Objects in Context. In *European Conference on Computer Vision* (pp. 740-755).

[6] WikiArt. (n.d.). WikiArt: Visual Art Encyclopedia. https://www.wikiart.org/

[7] DiffSynth-Studio. ImagePulse-StyleTransfer [Dataset]. ModelScope. https://www.modelscope.cn/datasets/DiffSynth-Studio/ImagePulse-StyleTransfer

---

## 📄 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{jing2025stylevar,
  title={StyleVAR: Controllable Image Style Transfer via Visual Autoregressive Modeling},
  author={Jing, Liqi and Zhang, Dingming and Li, Peinian},
  journal={Duke University Course Project},
  year={2025}
}
```

---

## 📧 Contact

For questions or collaboration opportunities, please contact:
- Liqi Jing: [liqi.jing@duke.edu](mailto:liqi.jing@duke.edu)
- GitHub Issues: [https://github.com/Senfier-LiqiJing/StyleVAR/issues](https://github.com/Senfier-LiqiJing/StyleVAR/issues)

---

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

**Acknowledgments**: This work builds upon the Visual Autoregressive Modeling (VAR) framework and utilizes the OmniStyle-150K dataset. We thank the authors for making their code and data publicly available
