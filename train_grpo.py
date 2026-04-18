"""
GRPO (Group Relative Policy Optimization) Training for StyleVAR with LoRA.

Memory-efficient design for 2x24GB GPUs:
  - cuda:0: StyleVAR (frozen base + LoRA adapters) + VAE (frozen) + Optimizer
  - cuda:1: Reward models (VGG-19, LPIPS-alex, TV)
  - Reference policy = same model with LoRA disabled (zero extra memory)

Dataset: unpaired content (coco2017) + style (wikiart) images.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 conda run -n var python train_grpo.py \
    --content_dir data/coco2017 \
    --style_dir data/wikiart \
    --var_ckpt  ckpt/style_var_d20_11_20_21.pth \
    --vae_ckpt  ckpt/vae_ch160v4096z32.pth
"""

import argparse
import gc
import glob
import math
import os
import random
import sys
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision.transforms import InterpolationMode, transforms

# ---------------------------------------------------------------------------
# Add project root to path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_vae_stylevar, VQVAE, StyleVAR
from models.helpers import sample_with_top_k_top_p_
from utils.data import normalize_01_into_pm1


# ============================= CLI =========================================
def parse_args():
    p = argparse.ArgumentParser("GRPO training for StyleVAR")
    # paths
    p.add_argument("--content_dir", type=str, default="data/coco2017/images/train2017",
                    help="Directory with content images (legacy unpaired mode)")
    p.add_argument("--style_dir",   type=str, default="data/wikiart",
                    help="Directory with style images (legacy unpaired mode)")
    # paired mode (recommended): uses (content, style, target) triplets
    p.add_argument("--paired_data", action="store_true",
                    help="Use paired (content, style, target) datasets for DreamSim reward")
    p.add_argument("--old_data_path", type=str, default="data/OmniStyle-150k",
                    help="OmniStyle triplet root (when --paired_data)")
    p.add_argument("--new_data_path", type=str, default="data/ImagePulse",
                    help="ImagePulse triplet root (when --paired_data)")
    p.add_argument("--vae_ckpt",  type=str, default="ckpt/vae_ch160v4096z32.pth")
    p.add_argument("--var_ckpt",  type=str, default="",
                    help="StyleVAR checkpoint (SFT result). Empty=use best/last from --sft_out_dir")
    p.add_argument("--sft_out_dir", type=str, default="Output",
                    help="SFT output dir to auto-find best/last checkpoint")
    p.add_argument("--out_dir",   type=str, default="./grpo_output")
    # GRPO
    p.add_argument("--G",            type=int,   default=8,     help="Group size (rollouts per prompt)")
    p.add_argument("--kl_coef",      type=float, default=0.01,  help="Initial KL penalty coefficient beta")
    p.add_argument("--kl_target",    type=float, default=0.2,   help="Target per-step raw KL for adaptive coefficient (0=fixed)")
    p.add_argument("--clip_eps",     type=float, default=0.2,   help="PPO-style clipping epsilon")
    p.add_argument("--log_ratio_clamp", type=float, default=10.0,
                    help="Clamp on |log(new_lp - old_lp)| before exp to prevent ratio explosion")
    p.add_argument("--kl_spike_threshold", type=float, default=5.0,
                    help="Per-G raw KL (kl_loss/kl_coef) above this skips the backward to prevent weight corruption")
    p.add_argument("--seed",         type=int,   default=42,
                    help="Global RNG seed (torch/np/random + per-step rollout seeds). Reproducible runs.")
    p.add_argument("--ref_update_every", type=int, default=0,
                    help="[LEGACY] Merge LoRA every N steps (0=disabled, use peak-triggered instead)")
    # Peak-triggered iterative merge (preferred over ref_update_every)
    p.add_argument("--merge_cooldown", type=int, default=0,
                    help="Minimum steps between NORMAL peak-triggered merges (0=disable peak-triggered merging)")
    p.add_argument("--merge_min_gain", type=float, default=0.02,
                    help="Required reward_ema improvement over last-merge baseline")
    p.add_argument("--merge_patience", type=int, default=30,
                    help="Steps to wait at peak before confirming merge trigger")
    p.add_argument("--merge_kl_threshold", type=float, default=0.5,
                    help="Emergency merge if raw KL exceeds this (0=disabled). Catches KL runaway before collapse.")
    p.add_argument("--merge_emergency_cooldown", type=int, default=50,
                    help="Minimum steps between EMERGENCY merges (separate from --merge_cooldown). "
                         "Emergency merges bypass the regular cooldown; this short cooldown just "
                         "prevents merge-thrashing within the same burst.")
    p.add_argument("--save_peak_lora", action="store_true",
                    help="Snapshot LoRA weights at every new peak; use snapshot for merge instead of current weights.")
    # reward weights (legacy LPIPS+Gram+SSIM path)
    p.add_argument("--lam_content",  type=float, default=1.0,   help="Weight for LPIPS content reward")
    p.add_argument("--lam_style",    type=float, default=1.0,   help="Weight for VGG Gram style reward")
    p.add_argument("--lam_tv",       type=float, default=0.01,  help="Weight for TV quality reward")
    p.add_argument("--lam_ssim",     type=float, default=0.5,   help="Weight for SSIM structure reward")
    # DreamSim reward (when --paired_data)
    p.add_argument("--use_dreamsim", action="store_true",
                    help="Use DreamSim(gen, target) as the sole reward (requires --paired_data)")
    p.add_argument("--dreamsim_scale", type=float, default=5.0,
                    help="Multiplier on -dreamsim distance for advantage scale")
    # CLIP reward (when --paired_data) — pilot 2 showed it ranks better than DreamSim
    p.add_argument("--use_clip_reward", action="store_true",
                    help="Use CLIP(gen, target) cosine similarity as the sole reward "
                         "(requires --paired_data). Mutually exclusive with --use_dreamsim.")
    p.add_argument("--clip_scale", type=float, default=5.0,
                    help="Multiplier on CLIP cosine similarity for advantage scale")
    p.add_argument("--clip_local_dir", type=str, default="",
                    help="Local HF snapshot of openai/clip-vit-base-patch32 (recommended to skip HF downloads)")
    # PANW (Per-Action Normalization Weighting)
    p.add_argument("--panw_alpha",   type=float, default=0.0,
                    help="PANW decay exponent (0=disabled). Weight per scale = 1/(h*w)^alpha. "
                         "Paper uses 0.6-0.8. Upweights coarse scales, downweights fine scales.")
    # training
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=6,     help="Condition pairs per step")
    p.add_argument("--lr",           type=float, default=2e-6)
    p.add_argument("--grad_clip",    type=float, default=10.0)
    p.add_argument("--save_every",   type=int,   default=200)
    p.add_argument("--log_every",    type=int,   default=1)
    # sampling
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--top_k",       type=int,   default=900)
    p.add_argument("--top_p",       type=float, default=0.96)
    # model
    p.add_argument("--depth",       type=int,   default=20)
    # training mode: full-param (default) or LoRA
    p.add_argument("--use_lora",    action="store_true", help="Use LoRA instead of full-param")
    p.add_argument("--lora_rank",   type=int,   default=256,   help="LoRA rank (only if --use_lora)")
    p.add_argument("--lora_alpha",  type=float, default=512.0, help="LoRA alpha (only if --use_lora)")
    # resume & logging
    p.add_argument("--resume",      type=str, default="",
                    help="Path to GRPO checkpoint to resume from (empty=auto-find latest)")
    p.add_argument("--wandb_project", type=str, default="StyleVAR")
    p.add_argument("--wandb_run_id",  type=str, default="")
    p.add_argument("--exp_name",      type=str, default="grpo_fullparam_v1")
    return p.parse_args()


# ======================== LoRA =============================================

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds LoRA adapters.

    Exposes a `.weight` property returning the effective weight
    (base + LoRA), so code that does `F.linear(x, module.weight, ...)`
    transparently gets the LoRA-augmented weight.
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # Store and freeze the original weight
        self.base_weight = base_linear.weight          # nn.Parameter, frozen below
        self.base_weight.requires_grad_(False)

        self.has_bias = base_linear.bias is not None
        if self.has_bias:
            self.base_bias = base_linear.bias           # nn.Parameter, frozen
            self.base_bias.requires_grad_(False)

        # LoRA adapters on same device/dtype as base weight
        dev, dtype = base_linear.weight.device, base_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self._enabled = True

    @property
    def weight(self):
        """Effective weight: W + B @ A * scaling.  Gradient flows through A,B."""
        if self._enabled:
            return self.base_weight + (self.lora_B @ self.lora_A) * self.scaling
        return self.base_weight

    @property
    def bias(self):
        return self.base_bias if self.has_bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def apply_lora(model: StyleVAR, rank: int, alpha: float) -> List[nn.Parameter]:
    """Freeze entire model, then inject LoRA into attention + FFN layers.

    Target layers per block:
      - mat_qkv_guide  (C -> 3C)   — Q/K/V for guide stream
      - mat_qkv_target (C -> 3C)   — Q/K/V for target stream
      - proj           (C -> C)    — attention output projection
      - fc1            (C -> 4C)   — FFN up projection
      - fc2            (4C -> C)   — FFN down projection

    Returns the list of trainable LoRA parameters.
    """
    # 1. Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    lora_params: List[nn.Parameter] = []
    for block in model.blocks:
        # Attention layers
        attn = block.attn
        for attr_name in ("mat_qkv_guide", "mat_qkv_target", "proj"):
            old_linear = getattr(attn, attr_name)
            new_module = LoRALinear(old_linear, rank, alpha)
            setattr(attn, attr_name, new_module)
            lora_params.extend([new_module.lora_A, new_module.lora_B])
        # FFN layers
        ffn = block.ffn
        for attr_name in ("fc1", "fc2"):
            old_linear = getattr(ffn, attr_name)
            new_module = LoRALinear(old_linear, rank, alpha)
            setattr(ffn, attr_name, new_module)
            lora_params.extend([new_module.lora_A, new_module.lora_B])

    total = sum(p.numel() for p in lora_params)
    print(f"[LoRA] Injected into {len(model.blocks)} blocks (attn+ffn), "
          f"{len(lora_params)} adapter tensors, {total:,} trainable params")
    return lora_params


def set_lora_enabled(model: StyleVAR, enabled: bool):
    """Toggle LoRA on/off.  Off = reference policy (base model only)."""
    for block in model.blocks:
        for attr_name in ("mat_qkv_guide", "mat_qkv_target", "proj"):
            module = getattr(block.attn, attr_name)
            if isinstance(module, LoRALinear):
                module._enabled = enabled
        for attr_name in ("fc1", "fc2"):
            module = getattr(block.ffn, attr_name)
            if isinstance(module, LoRALinear):
                module._enabled = enabled


def snapshot_lora(model: StyleVAR) -> dict:
    """Return a CPU copy of all LoRA parameter tensors, keyed by module id."""
    snap = {}
    for block in model.blocks:
        for attr_name in ("mat_qkv_guide", "mat_qkv_target", "proj"):
            m = getattr(block.attn, attr_name)
            if isinstance(m, LoRALinear):
                snap[id(m)] = (m.lora_A.detach().clone().cpu(),
                               m.lora_B.detach().clone().cpu())
        for attr_name in ("fc1", "fc2"):
            m = getattr(block.ffn, attr_name)
            if isinstance(m, LoRALinear):
                snap[id(m)] = (m.lora_A.detach().clone().cpu(),
                               m.lora_B.detach().clone().cpu())
    return snap


def restore_lora(model: StyleVAR, snap: dict):
    """Write snapshotted LoRA tensors back into the model (in place)."""
    for block in model.blocks:
        for attr_name in ("mat_qkv_guide", "mat_qkv_target", "proj"):
            m = getattr(block.attn, attr_name)
            if isinstance(m, LoRALinear) and id(m) in snap:
                A, B = snap[id(m)]
                with torch.no_grad():
                    m.lora_A.copy_(A.to(m.lora_A.device))
                    m.lora_B.copy_(B.to(m.lora_B.device))
        for attr_name in ("fc1", "fc2"):
            m = getattr(block.ffn, attr_name)
            if isinstance(m, LoRALinear) and id(m) in snap:
                A, B = snap[id(m)]
                with torch.no_grad():
                    m.lora_A.copy_(A.to(m.lora_A.device))
                    m.lora_B.copy_(B.to(m.lora_B.device))


def merge_lora_and_reinit(model: StyleVAR) -> List[nn.Parameter]:
    """Merge LoRA weights into base, reinitialize LoRA to zero.

    After this call:
      - base_weight contains the old (base + LoRA) effective weight
      - lora_A is re-initialized (kaiming), lora_B is zeroed
      - set_lora_enabled(False) gives the NEW reference (= old policy)
      - set_lora_enabled(True)  gives the NEW policy   (= old policy + fresh LoRA ≈ old policy)
      - KL between policy and reference starts at ~0
    Returns the new list of trainable LoRA parameters (for optimizer reset).
    """
    lora_params: List[nn.Parameter] = []
    for block in model.blocks:
        for attr_name in ("mat_qkv_guide", "mat_qkv_target", "proj"):
            module = getattr(block.attn, attr_name)
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.base_weight.add_((module.lora_B @ module.lora_A) * module.scaling)
                    nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                    module.lora_B.zero_()
                lora_params.extend([module.lora_A, module.lora_B])
        for attr_name in ("fc1", "fc2"):
            module = getattr(block.ffn, attr_name)
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    module.base_weight.add_((module.lora_B @ module.lora_A) * module.scaling)
                    nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                    module.lora_B.zero_()
                lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


# ======================== Dataset ==========================================

class UnpairedStyleContentDataset(Dataset):
    """
    Random pairing of content images (e.g. COCO) and style images (e.g. WikiArt).
    No target / ground-truth image needed — GRPO learns from reward signals.
    """
    EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp",
            "*.JPG", "*.JPEG", "*.PNG", "*.WEBP")

    def __init__(self, content_dir: str, style_dir: str, transform=None):
        self.content_files: List[str] = []
        self.style_files: List[str] = []
        for ext in self.EXTS:
            self.content_files.extend(
                glob.glob(os.path.join(content_dir, "**", ext), recursive=True))
            self.style_files.extend(
                glob.glob(os.path.join(style_dir, "**", ext), recursive=True))
        self.content_files.sort()
        self.style_files.sort()
        self.transform = transform
        assert len(self.content_files) > 0, f"No content images in {content_dir}"
        assert len(self.style_files) > 0,   f"No style images in {style_dir}"
        print(f"[Dataset] Content: {len(self.content_files)}, "
              f"Style: {len(self.style_files)}")

    def __len__(self):
        return len(self.content_files)

    def __getitem__(self, idx):
        content = Image.open(self.content_files[idx]).convert("RGB")
        style_idx = random.randint(0, len(self.style_files) - 1)
        style = Image.open(self.style_files[style_idx]).convert("RGB")
        if self.transform:
            content = self.transform(content)
            style = self.transform(style)
        # Return (style, content) — no target needed
        return style, content


# ======================== Reward Models ====================================

class VGGGramStyleReward(nn.Module):
    """Negative VGG-19 Gram-matrix MSE (style loss). Higher = more stylistic."""
    STYLE_LAYERS = ["0", "5", "10", "19", "28"]   # conv1_1 .. conv5_1

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        prev = 0
        for idx_str in self.STYLE_LAYERS:
            idx = int(idx_str) + 1
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:idx]))
            prev = idx
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _gram(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = x.view(B, C, H * W)
        return torch.bmm(feat, feat.transpose(1, 2)) / (C * H * W)

    @torch.no_grad()
    def forward(self, gen_01: torch.Tensor, style_01: torch.Tensor) -> torch.Tensor:
        gen   = (gen_01   - self.mean) / self.std
        style = (style_01 - self.mean) / self.std
        loss = gen.new_zeros(gen.shape[0])
        x_g, x_s = gen, style
        for sl in self.slices:
            x_g = sl(x_g)
            x_s = sl(x_s)
            loss += (self._gram(x_g) - self._gram(x_s)).square().mean(dim=(1, 2))
        return -loss      # negative so higher = better


class TVReward(nn.Module):
    """Negative Total-Variation reward (pure tensor math, no params)."""
    @torch.no_grad()
    def forward(self, img_01: torch.Tensor) -> torch.Tensor:
        dh = (img_01[:, :, 1:, :] - img_01[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
        dw = (img_01[:, :, :, 1:] - img_01[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
        return -(dh + dw)


class SSIMReward(nn.Module):
    """SSIM reward (pure tensor math, no learned params).
    Measures structural similarity between generated and content images.
    Returns per-sample SSIM in [0, 1], higher = better structure preservation.
    """
    def __init__(self, window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2):
        super().__init__()
        self.C1, self.C2 = C1, C2
        # Gaussian window
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)  # 2D
        window = window.expand(3, 1, window_size, window_size).contiguous()
        self.register_buffer("window", window)
        self.pad = window_size // 2

    @torch.no_grad()
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """img1, img2: (B, 3, H, W) in [0, 1]. Returns (B,) SSIM scores."""
        mu1 = F.conv2d(img1, self.window, padding=self.pad, groups=3)
        mu2 = F.conv2d(img2, self.window, padding=self.pad, groups=3)
        mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.pad, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.pad, groups=3) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, self.window, padding=self.pad, groups=3) - mu12
        ssim_map = ((2*mu12 + self.C1) * (2*sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        return ssim_map.mean(dim=(1, 2, 3))  # (B,)


def _load_lpips_alex(device: torch.device):
    """Load LPIPS (AlexNet) on *device*, frozen."""
    import lpips
    net = lpips.LPIPS(net="alex").to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class DreamSimReward(nn.Module):
    """DreamSim perceptual similarity reward.

    Returns negative DreamSim distance so higher = more similar.
    DreamSim is a learned perceptual metric (DINO-based) aligned with human
    judgment of holistic image similarity. Unlike LPIPS (low-level texture),
    DreamSim captures semantic/structural similarity.
    """

    def __init__(self, device: torch.device, dreamsim_type: str = "dino_vitb16",
                 cache_dir: str = None):
        super().__init__()
        from dreamsim import dreamsim
        import sys
        # Default cache_dir = <project_root>/ckpt/dreamsim (absolute path,
        # independent of CWD so it works regardless of where the script is run from).
        if cache_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(project_root, "ckpt", "dreamsim")
        cache_dir = os.path.abspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[DreamSim] cache_dir = {cache_dir}")
        # Verify cache contents before loading
        required = {
            "dino_vitb16": ["dino_vitb16_pretrain.pth", "dino_vitb16_single_lora"],
        }[dreamsim_type]
        missing = [r for r in required
                   if not os.path.exists(os.path.join(cache_dir, r))]
        if missing:
            print(f"[DreamSim] WARNING: cache missing {missing} — will attempt download")

        # HACK: DINO's hubconf imports `vision_transformer` which does
        # `from utils import trunc_normal_`. Python sys.modules already has
        # StyleVAR's `utils/` package cached, causing ImportError. Temporarily
        # evict StyleVAR's `utils` from sys.modules so DINO's local utils.py
        # gets loaded via torch.hub's sys.path injection. Restore after.
        saved_utils_modules = {}
        for k in list(sys.modules.keys()):
            if k == "utils" or k.startswith("utils."):
                saved_utils_modules[k] = sys.modules.pop(k)
        try:
            self.model, _ = dreamsim(pretrained=True, device=device,
                                      dreamsim_type=dreamsim_type,
                                      cache_dir=cache_dir)
        finally:
            # Remove any DINO-side 'utils' / 'vision_transformer' leftovers
            # to avoid collision with StyleVAR's utils on restore.
            for k in list(sys.modules.keys()):
                if k == "utils" or k == "vision_transformer" or k.startswith("utils."):
                    sys.modules.pop(k, None)
            sys.modules.update(saved_utils_modules)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self._target_size = 224

    @torch.no_grad()
    def forward(self, gen_01: torch.Tensor, target_01: torch.Tensor) -> torch.Tensor:
        """Both inputs: [B,3,H,W] in [0,1]. DreamSim handles its own normalization.
        Returns (B,) — higher = more similar (we negate the distance)."""
        if gen_01.shape[-1] != self._target_size:
            gen_01 = F.interpolate(gen_01, size=self._target_size,
                                    mode="bilinear", align_corners=False)
        if target_01.shape[-1] != self._target_size:
            target_01 = F.interpolate(target_01, size=self._target_size,
                                       mode="bilinear", align_corners=False)
        dist = self.model(gen_01, target_01)  # (B,) in [0, ~1]
        return -dist


class CLIPReward(nn.Module):
    """CLIP image-image cosine similarity reward.

    Returns cosine similarity in [-1, 1], higher = more similar. Use against
    the paired-data target (GT stylized output) for a semantic, stylization-robust
    reward signal. Pilot 2 showed CLIP(out, content) tracks CLIP(out, GT) with
    Spearman rho ~0.8 — much higher than DreamSim (0.64), making CLIP a more
    stable ranking metric for style-transfer candidates.

    Load priority: local HF snapshot > HF hub > open_clip.
    """
    def __init__(self, device: torch.device, local_dir: str = ""):
        super().__init__()
        self.backend = None
        if local_dir and os.path.isdir(local_dir):
            from transformers import CLIPModel
            self.model = CLIPModel.from_pretrained(local_dir).to(device).eval()
            self.backend = f"hf-local:{local_dir}"
        if self.backend is None:
            try:
                from transformers import CLIPModel
                self.model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32").to(device).eval()
                self.backend = "hf-hub"
            except Exception as e:
                print(f"[CLIPReward] HF hub failed: {e}")
        if self.backend is None:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai")
            self.model = model.to(device).eval()
            self.backend = "open_clip"
        print(f"[CLIPReward] backend={self.backend}")
        for p in self.parameters(): p.requires_grad_(False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.to(device)

    @torch.no_grad()
    def _encode(self, img_01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(img_01, size=224, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        if self.backend == "open_clip":
            feat = self.model.encode_image(x)
        else:
            vision_out = self.model.vision_model(pixel_values=x)
            pooled = (vision_out.pooler_output if hasattr(vision_out, "pooler_output")
                      else vision_out[1])
            feat = self.model.visual_projection(pooled)
        return F.normalize(feat.float(), dim=-1)

    @torch.no_grad()
    def forward(self, gen_01: torch.Tensor, target_01: torch.Tensor) -> torch.Tensor:
        """gen_01, target_01: (B,3,H,W) in [0,1]. Returns (B,) cosine in [-1,1]."""
        fg = self._encode(gen_01)
        ft = self._encode(target_01)
        return (fg * ft).sum(dim=-1)


# ======================== Model Building ===================================

def _build_model(args):
    """
    Build StyleVAR + VAE on cuda:0.
    Does NOT inject LoRA or set trainability — that's done in main().
    """
    DEV = torch.device("cuda:0")
    patch_nums = tuple(int(x) for x in "1_2_3_4_5_6_8_10_13_16".split("_"))

    vae, model = build_vae_stylevar(
        device=DEV, patch_nums=patch_nums,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        depth=args.depth, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
        style_enc_dim=512,
    )

    # Load VAE weights (frozen)
    vae_state = torch.load(args.vae_ckpt, map_location="cpu")
    vae.load_state_dict(vae_state, strict=True)
    print(f"[GRPO] VAE loaded from {args.vae_ckpt}")

    # Load StyleVAR checkpoint (SFT result)
    if args.var_ckpt and os.path.exists(args.var_ckpt):
        ckpt = torch.load(args.var_ckpt, map_location="cpu")
        if "trainer" in ckpt and "var_wo_ddp" in ckpt["trainer"]:
            st = ckpt["trainer"]["var_wo_ddp"]
        elif "model" in ckpt:
            st = ckpt["model"]
        else:
            st = ckpt

        # Safely load: STRICT first, then auto-bake LoRA wrappers if present.
        # DO NOT fall back to strict=False silently — that hid a major bug (v3
        # trained on random-init transformer blocks because v2 ckpt keys were
        # LoRA-wrapped and silently dropped).
        _load_var_state_strict(model, st, ckpt if isinstance(ckpt, dict) else {})
        print(f"[GRPO] StyleVAR loaded from {args.var_ckpt}")
    else:
        print("[GRPO] WARNING: No VAR checkpoint — training from scratch init")

    return model, vae, patch_nums


def _load_var_state_strict(model, st: dict, full_ckpt: dict):
    """Load a StyleVAR state_dict with strict=True, auto-baking LoRA wrappers.

    Handles three cases (all verified to be strict-equivalent after handling):
      1. Plain SFT state_dict (`blocks.X.*.weight`)  → strict=True directly.
      2. LoRA-wrapped full_state (has `.base_weight` / `.lora_A` / `.lora_B`)
         → bake delta into `.weight`, strip LoRA, then strict=True.
      3. LoRA-only state_dict (only `.lora_A` / `.lora_B`)
         → raise — this isn't a valid base-model ckpt.

    Any unexpected or missing keys after handling raise a RuntimeError.
    """
    has_base_weight = any(k.endswith(".base_weight") for k in st.keys())
    has_lora_A      = any(k.endswith(".lora_A")      for k in st.keys())
    has_plain_weight = any(
        k.endswith(".weight") and not (k.endswith(".base_weight")
                                       or ".lora_" in k)
        for k in st.keys()
    )

    # Case 3: only LoRA keys, no base → unusable as a base ckpt.
    if has_lora_A and not has_base_weight:
        raise RuntimeError(
            "[load_var] Refused: ckpt contains only LoRA-only keys "
            "(no base_weight / weight for transformer blocks). "
            "This would silently train on random-init transformer. "
            "Provide a full-state or plain-SFT ckpt, or merge the LoRA first."
        )

    # Case 2: LoRA-wrapped full_state → bake LoRA into plain weights.
    if has_base_weight:
        args_in_ckpt = full_ckpt.get("args", {}) if isinstance(full_ckpt, dict) else {}
        rank  = args_in_ckpt.get("lora_rank", 256)
        alpha = args_in_ckpt.get("lora_alpha", 512.0)
        scaling = alpha / rank
        print(f"[load_var] Detected LoRA-wrapped ckpt; baking delta "
              f"(rank={rank}, alpha={alpha}, scaling={scaling}) into plain weights.")
        prefixes = {k[:-len(".lora_A")] for k in st.keys() if k.endswith(".lora_A")}
        baked = {}
        n_baked = 0
        for k, v in st.items():
            matched = False
            for p in prefixes:
                if k == p + ".base_weight":
                    lA = st[p + ".lora_A"]; lB = st[p + ".lora_B"]
                    baked[p + ".weight"] = (v + (lB @ lA) * scaling).contiguous()
                    n_baked += 1; matched = True; break
                if k == p + ".base_bias":
                    baked[p + ".bias"] = v; matched = True; break
                if k == p + ".lora_A" or k == p + ".lora_B":
                    matched = True; break
            if not matched:
                baked[k] = v
        print(f"[load_var] Baked {n_baked} LoRA modules into plain weights.")
        st = baked

    # Strict load — will raise RuntimeError on ANY missing / unexpected key.
    model.load_state_dict(st, strict=True)


# ======================== Rollout (Autoregressive Sampling) ================

@torch.no_grad()
def rollout_generate(
    actor: StyleVAR,
    vae: VQVAE,
    style_pm1: torch.Tensor,    # (B,3,H,W) in [-1,1]
    content_pm1: torch.Tensor,  # (B,3,H,W) in [-1,1]
    G: int,
    temperature: float = 1.0,
    top_k: int = 900,
    top_p: float = 0.96,
    base_seed: int = None,      # if set: rollout g uses base_seed + g (reproducible)
) -> Tuple[List[List[torch.LongTensor]], torch.Tensor]:
    """
    Generate G rollouts per condition pair (no CFG).

    Returns:
        all_token_trajs: G lists, each containing 10 scale index tensors (B, pn^2)
        gen_images_01:   (B*G, 3, H, W) in [0,1]
    """
    DEV = style_pm1.device
    B = style_pm1.shape[0]
    patch_nums = actor.patch_nums
    quantize = vae.quantize

    # ---- Pre-compute deterministic conditioning (shared across G rollouts) ----
    sos = actor.feat_emb(
        actor.content_encoder(content_pm1).squeeze(-1).squeeze(-1))     # (B, C)
    cond_BD = actor.feat_emb(
        actor.style_encoder(style_pm1).squeeze(-1).squeeze(-1))         # (B, C)

    lvl_pos = actor.lvl_embed(actor.lvl_1L) + actor.pos_1LC             # (1, L, C)

    # VQ-VAE tokenise style & content once (deterministic)
    ms_style_idx  = vae.img_to_idxBl(style_pm1)
    ms_style_BlCv = quantize.msBllist_to_BlCv_list(ms_style_idx)
    ms_style_BlC  = [actor.word_embed(item) for item in ms_style_BlCv]

    ms_content_idx  = vae.img_to_idxBl(content_pm1)
    ms_content_BlCv = quantize.msBllist_to_BlCv_list(ms_content_idx)
    ms_content_BlC  = [actor.word_embed(item) for item in ms_content_BlCv]

    # Add level + position embeddings to style/content tokens
    cur_L = 0
    for idx_s, sBlC in enumerate(ms_style_BlC):
        pn = patch_nums[idx_s]
        ms_style_BlC[idx_s]   = sBlC + lvl_pos[:, cur_L:cur_L + pn**2]
        ms_content_BlC[idx_s] = ms_content_BlC[idx_s] + lvl_pos[:, cur_L:cur_L + pn**2]
        cur_L += pn**2

    # ---- Generate G rollouts ----
    all_token_trajs: List[List[torch.LongTensor]] = []
    all_images: List[torch.Tensor] = []

    for g_idx in range(G):
        rng = torch.Generator(device=DEV)
        if base_seed is None:
            rng.manual_seed(int(time.time() * 1000 + g_idx) % (2**31))
        else:
            rng.manual_seed((int(base_seed) + g_idx) % (2**31))

        # Init next_token_map from SOS
        next_token_map = (
            sos.unsqueeze(1).expand(B, actor.first_l, -1)
            + actor.pos_start.expand(B, actor.first_l, -1)
            + lvl_pos[:, :actor.first_l]
        )

        f_hat = sos.new_zeros(B, actor.Cvae, patch_nums[-1], patch_nums[-1])
        idx_per_scale: List[torch.LongTensor] = []

        # Enable KV caching (resets cache)
        for b in actor.blocks:
            b.attn.kv_caching(True)

        cur_L = 0
        for si, pn in enumerate(patch_nums):
            cur_L += pn * pn
            cond_BD_or_gss = actor.shared_ada_lin(cond_BD)
            x = next_token_map
            for b in actor.blocks:
                x = b(x=x, style=ms_style_BlC[si], content=ms_content_BlC[si],
                      cond_BD=cond_BD_or_gss, attn_bias=None,
                      alpha=actor.alpha_nums[si])

            logits_BlV = actor.get_logits(x, cond_BD)
            if temperature != 1.0:
                logits_BlV = logits_BlV / temperature

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV, rng=rng, top_k=top_k, top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            idx_per_scale.append(idx_Bl)

            h_BChw = quantize.embedding(idx_Bl)
            h_BChw = h_BChw.transpose(1, 2).reshape(B, actor.Cvae, pn, pn)
            f_hat, next_token_map = quantize.get_next_autoregressive_input(
                si, len(patch_nums), f_hat, h_BChw)

            if si != actor.num_stages_minus_1:
                next_token_map = (
                    next_token_map.view(B, actor.Cvae, -1).transpose(1, 2))
                next_pn = patch_nums[si + 1]
                next_token_map = (
                    actor.word_embed(next_token_map)
                    + lvl_pos[:, cur_L:cur_L + next_pn**2]
                )

        # Disable KV caching
        for b in actor.blocks:
            b.attn.kv_caching(False)

        gen_img = vae.fhat_to_img(f_hat).add_(1).mul_(0.5)   # [-1,1] -> [0,1]
        all_images.append(gen_img)
        all_token_trajs.append(idx_per_scale)

    gen_images_01 = torch.cat(all_images, dim=0)              # (B*G, 3, H, W)
    return all_token_trajs, gen_images_01


# ======================== Log-prob Computation ==============================

def compute_logprobs_single(
    model: StyleVAR,
    vae: VQVAE,
    idx_list: List[torch.LongTensor],   # one rollout: list of (B, pn^2)
    style_pm1: torch.Tensor,            # (B,3,H,W) in [-1,1]
    content_pm1: torch.Tensor,          # (B,3,H,W) in [-1,1]
    style_BLCvae: torch.Tensor,         # precomputed (B, L, Cvae)
    content_BLCvae: torch.Tensor,       # precomputed (B, L, Cvae)
) -> torch.Tensor:
    """
    Teacher-forcing forward pass for ONE rollout.
    Returns per-token log-probs: (B, L).
    """
    quantize = vae.quantize

    gt_BL = torch.cat(idx_list, dim=1)                          # (B, L)

    # Build teacher-forcing input from sampled tokens
    x_BLCv_wo_first_l = quantize.idxBl_to_var_input(idx_list)   # (B, L-first_l, Cvae)

    # Forward (teacher forcing with causal mask)
    logits_BLV = model(
        x_BLCv_wo_first_l, style_BLCvae, content_BLCvae,
        style_pm1, content_pm1,
    )                                                            # (B, L, V)

    log_probs = F.log_softmax(logits_BLV.float(), dim=-1)        # (B, L, V)
    chosen_lp = log_probs.gather(2, gt_BL.unsqueeze(2)).squeeze(2)  # (B, L)
    return chosen_lp


def precompute_style_content_features(vae: VQVAE, style_pm1, content_pm1):
    """Compute VQ-VAE multi-scale features for style/content (deterministic).
    Returns (style_BLCvae, content_BLCvae) each of shape (B, L, Cvae).
    """
    quantize = vae.quantize
    ms_style_idx   = vae.img_to_idxBl(style_pm1)
    style_BLCvae   = quantize.msBllist_to_BlCvae(ms_style_idx)
    ms_content_idx = vae.img_to_idxBl(content_pm1)
    content_BLCvae = quantize.msBllist_to_BlCvae(ms_content_idx)
    return style_BLCvae, content_BLCvae


# ======================== Reward Computation ================================

@torch.no_grad()
def compute_rewards(
    gen_images_01: torch.Tensor,     # (B*G, 3, H, W) in [0,1]
    style_img_01: torch.Tensor,      # (B, 3, H, W) in [0,1]
    content_img_01: torch.Tensor,    # (B, 3, H, W) in [0,1]
    G: int,
    lpips_net: nn.Module,
    vgg_gram: VGGGramStyleReward,
    tv_reward: TVReward,
    ssim_reward: SSIMReward,
    lam_content: float,
    lam_style: float,
    lam_tv: float,
    lam_ssim: float,
) -> torch.Tensor:
    """Compute composite reward R_i for each generated image.
    gen_images_01 layout: [g0_b0, g0_b1, ..., g0_bB, g1_b0, ..., g1_bB, ...]
    Returns (total_reward, r_content, r_style, r_tv, r_ssim), each (B*G,)."""
    BG = gen_images_01.shape[0]
    B = BG // G

    # Expand conditions: repeat B-block G times to match gen layout
    style_exp   = style_img_01.repeat(G, 1, 1, 1)      # (B*G, 3, H, W)
    content_exp = content_img_01.repeat(G, 1, 1, 1)

    # Content reward: negative LPIPS (expects [-1,1])
    gen_pm1     = gen_images_01 * 2 - 1
    content_pm1 = content_exp * 2 - 1
    r_content   = -lpips_net(gen_pm1, content_pm1).view(BG)

    # Style reward: negative VGG Gram MSE (expects [0,1])
    r_style = vgg_gram(gen_images_01, style_exp)

    # Quality reward: negative TV
    r_tv = tv_reward(gen_images_01)

    # Structure reward: SSIM (generated vs content, already in [0,1])
    r_ssim = ssim_reward(gen_images_01, content_exp)

    total = (lam_content * r_content + lam_style * r_style +
             lam_tv * r_tv + lam_ssim * r_ssim)
    return total, r_content, r_style, r_tv, r_ssim


def build_panw_weights(patch_nums: tuple, alpha: float, device: torch.device) -> torch.Tensor:
    """Build PANW per-token weight vector.

    For each scale t with grid size (pn × pn), all tokens get weight 1/(pn²)^α.
    Returns (1, L) tensor where L = sum(pn² for pn in patch_nums) = 680.
    Weights are normalized so they sum to 1 (making weighted mean easy).
    """
    weights = []
    for pn in patch_nums:
        num_tokens = pn * pn
        w = 1.0 / (num_tokens ** alpha)
        weights.extend([w] * num_tokens)
    w = torch.tensor(weights, device=device, dtype=torch.float32)
    w = w / w.sum()  # normalize so weighted mean = (loss * w).sum()
    return w.unsqueeze(0)  # (1, L)


def compute_group_advantages(rewards: torch.Tensor, G: int) -> torch.Tensor:
    """Group-relative advantage: A_i = (R_i - mu) / (sigma + eps).
    rewards layout: [g0_b0, ..., g0_bB, g1_b0, ..., g1_bB, ...] shape (B*G,)
    Returns same layout (B*G,)."""
    B = rewards.shape[0] // G
    # Reshape to (G, B) then transpose to (B, G) for group-wise normalization
    R = rewards.view(G, B).T                   # (B, G)
    mu    = R.mean(dim=1, keepdim=True)
    sigma = R.std(dim=1, keepdim=True)
    A = (R - mu) / (sigma.clamp(min=1e-2) + 1e-8)  # (B, G)
    return A.T.contiguous().view(B * G)        # back to (G, B) -> flatten


# ======================== Checkpoint helpers =================================

def _resolve_sft_checkpoint(args):
    """Find the best SFT checkpoint to initialize from."""
    if args.var_ckpt and os.path.exists(args.var_ckpt):
        return args.var_ckpt
    # Auto-search in sft_out_dir
    for name in ("ar-ckpt-best.pth", "ar-ckpt-last.pth"):
        p = os.path.join(args.sft_out_dir, name)
        if os.path.exists(p):
            return p
    # Fallback to ckpt dir
    p = os.path.join("ckpt", "style_var_d20_11_20_21.pth")
    if os.path.exists(p):
        return p
    raise FileNotFoundError("No StyleVAR checkpoint found. Specify --var_ckpt")


def _find_latest_grpo_ckpt(out_dir):
    """Find the latest GRPO checkpoint in out_dir."""
    ckpts = sorted(glob.glob(os.path.join(out_dir, "grpo_*.pth")),
                   key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


def _save_grpo_ckpt(model, optimizer, scaler, global_step, epoch, args, suffix,
                    full_state=False):
    """Save GRPO checkpoint.

    full_state=True saves entire model (needed after merge_lora_and_reinit,
    because base_weight has changed and LoRA-only keys are zeroed).
    """
    ckpt_path = os.path.join(args.out_dir, f"grpo_{suffix}.pth")
    if args.use_lora and not full_state:
        model_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    else:
        model_state = model.state_dict()
    torch.save({
        "step": global_step,
        "epoch": epoch,
        "use_lora": args.use_lora,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else {},
        "args": vars(args),
    }, ckpt_path)
    return ckpt_path


# ======================== Main Training Loop ===============================

def main():
    import copy

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Global RNG seeding (reproducibility) ----
    random.seed(args.seed)
    import numpy as _np
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"[GRPO] RNG seeded with seed={args.seed}")

    DEV = torch.device("cuda:0")

    # ---- Resolve SFT checkpoint ----
    args.var_ckpt = _resolve_sft_checkpoint(args)

    mode = "LoRA" if args.use_lora else "Full-param"
    print("=" * 60)
    print(f"  GRPO Training for StyleVAR  ({mode})")
    print(f"  Device: {DEV}")
    print(f"  SFT ckpt: {args.var_ckpt}")
    print(f"  G={args.G}, bs={args.batch_size}, lr={args.lr}")
    if args.use_lora:
        print(f"  LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  kl_coef={args.kl_coef}, clip_eps={args.clip_eps}")
    print("=" * 60)

    # ---- 1. Build model ----
    model, vae, patch_nums = _build_model(args)

    # Freeze VAE (CRITICAL: quant.py uses .transpose_() which crashes backward
    # if embedding.weight.requires_grad is True)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    if args.use_lora:
        # LoRA mode: inject adapters, ref policy = LoRA disabled
        lora_params = apply_lora(model, args.lora_rank, args.lora_alpha)
        train_params = lora_params
        ref_model = None  # use toggle
    else:
        # Full-param mode: all params trainable, deepcopy ref model
        train_params = [p for p in model.parameters() if p.requires_grad]
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        print(f"[GRPO] Ref model created (frozen copy, {sum(p.numel() for p in ref_model.parameters())/1e6:.0f}M params)")

    total_train = sum(p.numel() for p in train_params)
    print(f"[GRPO] Trainable params: {total_train:,} ({total_train/721e6*100:.1f}%)")

    # ---- 2. Reward models ----
    dreamsim_net = None
    clip_net = None
    lpips_net = vgg_gram = tv_reward_fn = ssim_reward_fn = None
    assert not (args.use_dreamsim and args.use_clip_reward), \
        "Cannot enable both --use_dreamsim and --use_clip_reward; pick one."
    if args.use_dreamsim:
        assert args.paired_data, "--use_dreamsim requires --paired_data"
        dreamsim_net = DreamSimReward(device=DEV)
        print(f"[GRPO] DreamSim reward loaded on {DEV}")
    elif args.use_clip_reward:
        assert args.paired_data, "--use_clip_reward requires --paired_data"
        clip_net = CLIPReward(device=DEV, local_dir=args.clip_local_dir)
        print(f"[GRPO] CLIP reward loaded on {DEV}")
    else:
        lpips_net = _load_lpips_alex(DEV)
        vgg_gram  = VGGGramStyleReward().to(DEV)
        tv_reward_fn = TVReward().to(DEV)
        ssim_reward_fn = SSIMReward().to(DEV)
        print(f"[GRPO] Legacy reward models (LPIPS+Gram+SSIM+TV) loaded on {DEV}")

    # ---- PANW weights ----
    panw_weights = None
    if args.panw_alpha > 0:
        patch_nums = tuple(int(x) for x in "1_2_3_4_5_6_8_10_13_16".split("_"))
        panw_weights = build_panw_weights(patch_nums, args.panw_alpha, DEV)
        print(f"[PANW] alpha={args.panw_alpha}, weights per scale: "
              + ", ".join(f"{pn}x{pn}={panw_weights[0, sum(p**2 for p in patch_nums[:i])].item():.4f}"
                          for i, pn in enumerate(patch_nums)))

    # ---- 3. Optimizer ----
    optimizer = torch.optim.AdamW(train_params, lr=args.lr,
                                   betas=(0.9, 0.95), weight_decay=0.01)
    scaler = None  # no mixed precision — full fp32 for numerical stability

    # ---- 4. Resume ----
    start_epoch = 0
    global_step = 0
    resume_path = args.resume
    if not resume_path:
        resume_path = _find_latest_grpo_ckpt(args.out_dir)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        model_sd = model.state_dict()
        model_sd.update(ckpt["model"])
        model.load_state_dict(model_sd)
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler and "scaler" in ckpt and ckpt["scaler"]:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        print(f"[GRPO] Resumed from {resume_path} (epoch={start_epoch}, step={global_step})")
    else:
        print("[GRPO] Starting fresh")

    # ---- 5. wandb ----
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            id=args.wandb_run_id or None,
            config=vars(args),
            resume="allow",
        )
        print(f"[wandb] {wandb_run.url}")
    except Exception as e:
        print(f"[wandb] Failed: {e}, continuing without wandb")

    # ---- 6. Dataset ----
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])
    if args.paired_data:
        # Reuse SFT's build_concat_dataset: returns (target, style, content)
        from utils.data import build_concat_dataset
        _, dataset, _ = build_concat_dataset(
            old_data_path=args.old_data_path,
            new_data_path=args.new_data_path,
            new_data_tar_dir="",
            final_reso=256,
        )
        paired = True
    else:
        dataset = UnpairedStyleContentDataset(
            content_dir=args.content_dir, style_dir=args.style_dir,
            transform=transform)
        paired = False
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    print(f"[GRPO] Dataset: {len(dataset)} samples, bs={args.batch_size}, paired={paired}")
    print(f"[GRPO] Steps/epoch: {len(dataloader)}, total: {len(dataloader)*args.epochs}")

    # ---- 7. Training loop ----
    best_reward_ema = -float("inf")
    reward_ema = None
    reward_ema_beta = 0.99
    kl_coef = args.kl_coef  # fixed coefficient
    has_merged = False  # track if LoRA has been merged (changes save strategy)

    # ---- Peak-triggered merge state (only if merge_cooldown > 0) ----
    ref_reward_ema = None         # reward_ema at time of last merge (or initial baseline)
    peak_since_merge = -float('inf')
    steps_at_peak = 0
    steps_since_merge = 0
    merge_count = 0
    peak_lora_snapshot = None     # CPU snapshot of LoRA weights at last peak (if save_peak_lora)

    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        for batch_idx, batch in enumerate(dataloader):
            t0 = time.time()
            # Unpack: paired=(target, style, content), unpaired=(style, content)
            if paired:
                target_pm1, style_pm1, content_pm1 = batch
                target_dev = target_pm1.to(DEV, non_blocking=True)
            else:
                style_pm1, content_pm1 = batch
                target_dev = None
            B = style_pm1.shape[0]

            style_dev   = style_pm1.to(DEV, non_blocking=True)
            content_dev = content_pm1.to(DEV, non_blocking=True)

            # ==============================================================
            # STEP 1: ROLLOUT
            # ==============================================================
            model.eval()
            if args.use_lora:
                set_lora_enabled(model, True)
            # Deterministic per-step rollout seed (still varies across steps / g_idx)
            rollout_base_seed = (args.seed * 1_000_003
                                  + global_step * args.G) % (2**31)
            token_trajs, gen_images_01 = rollout_generate(
                model, vae, style_dev, content_dev,
                G=args.G, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                base_seed=rollout_base_seed,
            )

            # ==============================================================
            # STEP 2: OLD LOG-PROBS
            # ==============================================================
            with torch.no_grad():
                style_feat, content_feat = precompute_style_content_features(
                    vae, style_dev, content_dev)
                old_logprobs_list = []
                for g in range(args.G):
                    lp = compute_logprobs_single(
                        model, vae, token_trajs[g],
                        style_dev, content_dev,
                        style_feat, content_feat)
                    old_logprobs_list.append(lp)
                old_logprobs = torch.cat(old_logprobs_list, dim=0)

            # ==============================================================
            # STEP 3: REFERENCE LOG-PROBS
            # ==============================================================
            with torch.no_grad():
                if args.use_lora:
                    set_lora_enabled(model, False)
                    ref_fwd_model = model
                else:
                    ref_fwd_model = ref_model

                ref_logprobs_list = []
                for g in range(args.G):
                    lp = compute_logprobs_single(
                        ref_fwd_model, vae, token_trajs[g],
                        style_dev, content_dev,
                        style_feat, content_feat)
                    ref_logprobs_list.append(lp)
                ref_logprobs = torch.cat(ref_logprobs_list, dim=0)

                if args.use_lora:
                    set_lora_enabled(model, True)

            # ==============================================================
            # STEP 4: REWARDS
            # ==============================================================
            style_01   = (style_dev + 1) * 0.5    # [-1,1] -> [0,1], no in-place
            content_01 = (content_dev + 1) * 0.5

            r_dream_mean = 0.0  # unscaled raw -dreamsim distance (for logging)
            r_clip_mean  = 0.0  # unscaled raw CLIP cosine similarity (for logging)
            if args.use_dreamsim:
                # DreamSim(gen, target) as sole reward
                target_01 = (target_dev + 1) * 0.5
                target_exp = target_01.repeat(args.G, 1, 1, 1)  # (B*G, 3, H, W)
                r_dream = dreamsim_net(gen_images_01, target_exp)  # (B*G,) in [-1, 0]
                rewards = r_dream * args.dreamsim_scale
                r_dream_mean = r_dream.mean().item()  # raw, unscaled
                r_content = r_style = r_tv = r_ssim = torch.zeros_like(rewards)
                del target_exp, target_01
            elif args.use_clip_reward:
                # CLIP(gen, target) cosine similarity as sole reward
                target_01 = (target_dev + 1) * 0.5
                target_exp = target_01.repeat(args.G, 1, 1, 1)
                r_clip = clip_net(gen_images_01, target_exp)  # (B*G,) in [-1, 1]
                rewards = r_clip * args.clip_scale
                r_clip_mean = r_clip.mean().item()
                r_content = r_style = r_tv = r_ssim = torch.zeros_like(rewards)
                del target_exp, target_01
            else:
                rewards, r_content, r_style, r_tv, r_ssim = compute_rewards(
                    gen_images_01, style_01, content_01, G=args.G,
                    lpips_net=lpips_net, vgg_gram=vgg_gram, tv_reward=tv_reward_fn,
                    ssim_reward=ssim_reward_fn,
                    lam_content=args.lam_content, lam_style=args.lam_style,
                    lam_tv=args.lam_tv, lam_ssim=args.lam_ssim,
                )
            advantages = compute_group_advantages(rewards, args.G)
            rewards_mean = rewards.mean().item()
            r_content_mean = r_content.mean().item()
            r_style_mean = r_style.mean().item()
            r_tv_mean = r_tv.mean().item()
            r_ssim_mean = r_ssim.mean().item()
            if reward_ema is None:
                reward_ema = rewards_mean
            else:
                reward_ema = reward_ema_beta * reward_ema + (1 - reward_ema_beta) * rewards_mean

            # ---- Peak-triggered merge: update tracking state ----
            if args.merge_cooldown > 0:
                if ref_reward_ema is None:
                    ref_reward_ema = reward_ema   # initial baseline = first observed reward
                steps_since_merge += 1
                if reward_ema > peak_since_merge:
                    peak_since_merge = reward_ema
                    steps_at_peak = 0              # new peak, reset patience counter
                    # Snapshot LoRA weights at this peak (before any degradation)
                    if args.save_peak_lora and args.use_lora:
                        peak_lora_snapshot = snapshot_lora(model)
                else:
                    steps_at_peak += 1

            # Debug: reward/advantage diagnostics on first 3 steps
            if global_step <= 2:
                print(f"  [DEBUG step={global_step}] rewards: {rewards[:8].tolist()}", flush=True)
                print(f"  [DEBUG step={global_step}] advantages: {advantages[:8].tolist()}", flush=True)
                print(f"  [DEBUG step={global_step}] style_dev range: [{style_dev.min():.2f}, {style_dev.max():.2f}]", flush=True)
                print(f"  [DEBUG step={global_step}] content_dev range: [{content_dev.min():.2f}, {content_dev.max():.2f}]", flush=True)

            del gen_images_01, style_01, content_01
            torch.cuda.empty_cache()

            # ==============================================================
            # STEP 5: POLICY UPDATE
            # ==============================================================
            # NOTE: keep model.eval() — do NOT call model.train()!
            # StyleVAR has BatchNorm in ResNet encoders. train() mode uses
            # batch statistics which differ from eval()'s running statistics,
            # causing old_lp != new_lp even before any gradient update.
            # eval() mode is fine for gradient computation (only affects BN/Dropout).
            optimizer.zero_grad()

            total_policy_loss = 0.0
            total_kl_loss = 0.0

            for g in range(args.G):
                new_lp = compute_logprobs_single(
                    model, vae, token_trajs[g],
                    style_dev, content_dev,
                    style_feat, content_feat)

                old_lp = old_logprobs[g*B:(g+1)*B].detach()
                ref_lp = ref_logprobs[g*B:(g+1)*B].detach()
                adv    = advantages[g*B:(g+1)*B].detach()

                # Clamp log_ratio before exp to prevent ratio explosion when
                # new_lp - old_lp is large. PPO clipping only bounds surr2;
                # surr1 for negative advantages is unclipped and can blow up.
                log_ratio = (new_lp.float() - old_lp.float()).clamp(
                    -args.log_ratio_clamp, args.log_ratio_clamp)
                ratio = torch.exp(log_ratio)

                # Debug: print diagnostics on first 3 steps
                if global_step <= 2 and g == 0:
                    print(f"  [DEBUG step={global_step} g={g}]", flush=True)
                    print(f"    ratio: mean={ratio.mean():.6f} std={ratio.std():.6f} "
                          f"min={ratio.min():.6f} max={ratio.max():.6f}", flush=True)
                    print(f"    log_ratio: mean={log_ratio.mean():.6f} abs_max={log_ratio.abs().max():.6f}", flush=True)
                    print(f"    new_lp: mean={new_lp.mean():.4f} std={new_lp.std():.4f}", flush=True)
                    print(f"    old_lp: mean={old_lp.mean():.4f} std={old_lp.std():.4f}", flush=True)
                    print(f"    ref_lp: mean={ref_lp.mean():.4f} std={ref_lp.std():.4f}", flush=True)
                    print(f"    old==new match: {(new_lp - old_lp).abs().max():.8f}", flush=True)
                    print(f"    old==ref match: {(old_lp - ref_lp).abs().max():.8f}", flush=True)
                    print(f"    advantage: mean={adv.mean():.4f} std={adv.std():.4f} "
                          f"min={adv.min():.4f} max={adv.max():.4f}", flush=True)
                    print(f"    mem: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
                adv_exp = adv.unsqueeze(1).expand_as(ratio)

                surr1 = ratio * adv_exp
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps,
                                    1.0 + args.clip_eps) * adv_exp
                per_token_policy = -torch.min(surr1, surr2)   # (B, L)

                # Non-negative KL estimator (Schulman): (ratio - 1) - log(ratio) >= 0
                log_ratio_kl = (new_lp.float() - ref_lp.float()).clamp(-10, 10)
                per_token_kl = kl_coef * (torch.exp(log_ratio_kl) - 1 - log_ratio_kl)  # (B, L)

                # PANW: weighted mean over tokens (upweights coarse scales)
                if panw_weights is not None:
                    policy_loss = (per_token_policy * panw_weights).sum(dim=1).mean()
                    kl_loss = (per_token_kl * panw_weights).sum(dim=1).mean()
                else:
                    policy_loss = per_token_policy.mean()
                    kl_loss = per_token_kl.mean()

                loss = (policy_loss + kl_loss) / args.G

                # Safety: skip if loss is NaN/Inf OR KL spike to prevent weight corruption.
                # kl_loss has kl_coef baked in (per Schulman k3 line above); divide it out
                # to compare against a scale-invariant threshold on the raw KL estimate.
                raw_kl_g = kl_loss.item() / max(kl_coef, 1e-8)
                kl_spike = raw_kl_g > args.kl_spike_threshold
                if torch.isfinite(loss) and not kl_spike:
                    loss.backward()
                    total_policy_loss += policy_loss.item() / args.G
                    total_kl_loss     += kl_loss.item() / args.G
                else:
                    reason = ("NaN/Inf" if not torch.isfinite(loss)
                              else f"KL spike (raw_kl={raw_kl_g:.2f} > {args.kl_spike_threshold})")
                    print(f"  [WARN] Skipping {reason} at g={g}", flush=True)
                    del new_lp, log_ratio, ratio, adv_exp, surr1, surr2
                    del log_ratio_kl, policy_loss, kl_loss, loss
                    optimizer.zero_grad()  # discard any accumulated grads
                    torch.cuda.empty_cache()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                train_params, max_norm=args.grad_clip)
            optimizer.step()

            # NOTE: kl_coef is fixed at args.kl_coef (0.02).
            # Adaptive KL failed in all variants:
            #   - Bidirectional: collapsed to 0.0009 (Run 15)
            #   - Upward-only: ratcheted to 0.22, killed learning (Run 16)
            # Fixed kl_coef + KL spike skip (>0.5) is sufficient.

            # ---- Iterative reference update ----
            # Mode 1: peak-triggered (preferred). Only merges when policy is
            #         provably better than last merge baseline AND has plateaued.
            # Mode 2: KL emergency trigger. Force merge if raw KL exceeds threshold
            #         (indicates runaway drift — restore peak snapshot if available).
            # Mode 3: fixed interval (legacy, unsafe — can ratchet reward down).
            should_merge = False
            merge_reason = ""
            use_peak_snapshot = False

            # Compute raw KL for emergency trigger check
            raw_kl = total_kl_loss / max(kl_coef, 1e-8) if total_kl_loss > 0 else 0.0

            if args.merge_cooldown > 0 and ref_reward_ema is not None:
                cooldown_ok = steps_since_merge >= args.merge_cooldown
                gain_ok = (peak_since_merge - ref_reward_ema) >= args.merge_min_gain
                patience_ok = steps_at_peak >= args.merge_patience
                # Normal peak-triggered merge
                if cooldown_ok and gain_ok and patience_ok:
                    should_merge = True
                    use_peak_snapshot = (peak_lora_snapshot is not None)
                    merge_reason = (
                        f"peak={peak_since_merge:.4f} ref={ref_reward_ema:.4f} "
                        f"gain={peak_since_merge-ref_reward_ema:+.4f} "
                        f"steps_at_peak={steps_at_peak}"
                    )
                # KL emergency — BYPASSES the regular cooldown (that's the
                # whole point of emergency). Uses a separate short cooldown to
                # prevent merge-thrashing in a single KL burst.
                # Does NOT require gain_ok: whole point is to rescue even when
                # model never improved (e.g. after resuming from a drifted ckpt).
                elif (args.merge_kl_threshold > 0
                      and raw_kl > args.merge_kl_threshold
                      and steps_since_merge >= args.merge_emergency_cooldown
                      and peak_lora_snapshot is not None):
                    should_merge = True
                    use_peak_snapshot = True
                    merge_reason = (
                        f"KL EMERGENCY: raw_kl={raw_kl:.3f} > threshold={args.merge_kl_threshold} "
                        f"(steps_since_merge={steps_since_merge} >= emergency_cooldown="
                        f"{args.merge_emergency_cooldown}) — restoring peak snapshot "
                        f"(peak={peak_since_merge:.4f})"
                    )
            elif args.ref_update_every > 0 and (global_step + 1) % args.ref_update_every == 0:
                should_merge = True
                merge_reason = f"fixed interval (every {args.ref_update_every})"

            if should_merge:
                print(f"  [MERGE] step={global_step+1}  {merge_reason}", flush=True)
                # If we have a peak snapshot, restore it before merging (so the
                # base_weight captures the PEAK policy, not the current degraded one)
                if use_peak_snapshot and peak_lora_snapshot is not None:
                    restore_lora(model, peak_lora_snapshot)
                    print(f"  [MERGE] restored LoRA from peak snapshot", flush=True)
                # Save full model state BEFORE merge (the policy being captured as new ref)
                _save_grpo_ckpt(model, optimizer, scaler, global_step + 1, epoch, args,
                                f"pre_merge_step{global_step+1}", full_state=True)
                train_params = merge_lora_and_reinit(model)
                optimizer = torch.optim.AdamW(train_params, lr=args.lr,
                                               betas=(0.9, 0.95), weight_decay=0.01)
                _save_grpo_ckpt(model, optimizer, scaler, global_step + 1, epoch, args,
                                "latest", full_state=True)
                has_merged = True
                merge_count += 1
                # Reset peak-tracking: new baseline = peak we just captured
                ref_reward_ema = peak_since_merge
                peak_since_merge = -float('inf')
                steps_at_peak = 0
                steps_since_merge = 0
                peak_lora_snapshot = None   # invalidated after merge
                print(f"  [MERGE] done. new ref_reward={ref_reward_ema:.4f}  merge_count={merge_count}", flush=True)

            # ---- Cleanup ----
            del old_logprobs, ref_logprobs, advantages, rewards
            del style_feat, content_feat, token_trajs
            gc.collect()
            torch.cuda.empty_cache()

            global_step += 1
            dt = time.time() - t0

            # ---- Logging ----
            if global_step % args.log_every == 0:
                mem = torch.cuda.max_memory_allocated(DEV) / 1e9
                if args.use_dreamsim:
                    reward_detail = f"[dream={r_dream_mean:.4f}]"
                elif args.use_clip_reward:
                    reward_detail = f"[clip_sim={r_clip_mean:.4f}]"
                else:
                    reward_detail = (f"[c={r_content_mean:.4f} s={r_style_mean:.4f} "
                                     f"ssim={r_ssim_mean:.4f} tv={r_tv_mean:.4f}]")
                print(
                    f"[Ep {epoch+1}/{args.epochs}] "
                    f"Step {global_step:5d} | "
                    f"loss={total_policy_loss + total_kl_loss:.4f} "
                    f"policy={total_policy_loss:.4f} "
                    f"kl={total_kl_loss:.4f} kl_coef={kl_coef:.4f} | "
                    f"R={rewards_mean:.4f} R_ema={reward_ema:.4f} "
                    f"{reward_detail} | "
                    f"grad={grad_norm:.3f} | "
                    f"mem={mem:.1f}G | "
                    f"{dt:.1f}s/step"
                )
                if wandb_run is not None:
                    wandb_log = {
                        "grpo/policy_loss": total_policy_loss,
                        "grpo/kl_loss": total_kl_loss,
                        "grpo/total_loss": total_policy_loss + total_kl_loss,
                        "grpo/reward_mean": rewards_mean,
                        "grpo/reward_ema": reward_ema,
                        "grpo/grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                        "grpo/kl_coef": kl_coef,
                        "grpo/lr": optimizer.param_groups[0]["lr"],
                        "grpo/mem_gb": mem,
                        "merge/ref_reward_ema": ref_reward_ema if ref_reward_ema is not None else 0.0,
                        "merge/peak_since_merge": peak_since_merge if peak_since_merge > -float('inf') else 0.0,
                        "merge/steps_at_peak": steps_at_peak,
                        "merge/steps_since_merge": steps_since_merge,
                        "merge/count": merge_count,
                    }
                    if args.use_dreamsim:
                        wandb_log["reward/dreamsim (neg distance)"] = r_dream_mean
                    elif args.use_clip_reward:
                        wandb_log["reward/clip_sim (cosine)"] = r_clip_mean
                    else:
                        wandb_log["reward/content (neg LPIPS)"] = r_content_mean
                        wandb_log["reward/style (neg Gram)"] = r_style_mean
                        wandb_log["reward/ssim (structure)"] = r_ssim_mean
                        wandb_log["reward/tv (neg TV)"] = r_tv_mean
                    wandb_run.log(wandb_log, step=global_step)

            # ---- Rolling checkpoint (5 slots) ----
            if global_step % args.save_every == 0:
                roll_idx = (global_step // args.save_every) % 5
                p = _save_grpo_ckpt(model, optimizer, scaler, global_step, epoch, args,
                                     f"roll{roll_idx}", full_state=has_merged)
                print(f"[ckpt] {os.path.basename(p)} (step={global_step})")
                _save_grpo_ckpt(model, optimizer, scaler, global_step, epoch, args,
                                "latest", full_state=has_merged)
                if reward_ema > best_reward_ema:
                    best_reward_ema = reward_ema
                    _save_grpo_ckpt(model, optimizer, scaler, global_step, epoch, args,
                                    "best", full_state=has_merged)
                    print(f"[ckpt] best updated (R_ema={reward_ema:.4f})")

        # ---- Epoch end ----
        ep_time = time.time() - t_epoch
        print(f"[Ep {epoch+1}] finished in {ep_time/60:.1f}min")
        _save_grpo_ckpt(model, optimizer, scaler, global_step, epoch + 1, args,
                         f"ep{epoch+1}", full_state=has_merged)
        _save_grpo_ckpt(model, optimizer, scaler, global_step, epoch + 1, args,
                         "latest", full_state=has_merged)

    # ---- Final save ----
    p = _save_grpo_ckpt(model, optimizer, scaler, global_step, args.epochs, args,
                         "final", full_state=has_merged)
    print(f"[GRPO] Final checkpoint: {p}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
