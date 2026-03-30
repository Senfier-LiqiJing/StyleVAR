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
                    help="Directory with content images (recursive search)")
    p.add_argument("--style_dir",   type=str, default="data/wikiart",
                    help="Directory with style images (recursive search)")
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
    # reward weights
    p.add_argument("--lam_content",  type=float, default=1.0,   help="Weight for LPIPS content reward")
    p.add_argument("--lam_style",    type=float, default=1.0,   help="Weight for VGG Gram style reward")
    p.add_argument("--lam_tv",       type=float, default=0.01,  help="Weight for TV quality reward")
    p.add_argument("--lam_ssim",     type=float, default=0.5,   help="Weight for SSIM structure reward")
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
        model.load_state_dict(st, strict=False)
        print(f"[GRPO] StyleVAR loaded from {args.var_ckpt}")
    else:
        print("[GRPO] WARNING: No VAR checkpoint — training from scratch init")

    return model, vae, patch_nums


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
        rng.manual_seed(int(time.time() * 1000 + g_idx) % (2**31))

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


def _save_grpo_ckpt(model, optimizer, scaler, global_step, epoch, args, suffix):
    """Save GRPO checkpoint."""
    ckpt_path = os.path.join(args.out_dir, f"grpo_{suffix}.pth")
    if args.use_lora:
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
    lpips_net = _load_lpips_alex(DEV)
    vgg_gram  = VGGGramStyleReward().to(DEV)
    tv_reward_fn = TVReward().to(DEV)
    ssim_reward_fn = SSIMReward().to(DEV)
    print(f"[GRPO] Reward models loaded on {DEV}")

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
    dataset = UnpairedStyleContentDataset(
        content_dir=args.content_dir, style_dir=args.style_dir,
        transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    print(f"[GRPO] Dataset: {len(dataset)} samples, bs={args.batch_size}")
    print(f"[GRPO] Steps/epoch: {len(dataloader)}, total: {len(dataloader)*args.epochs}")

    # ---- 7. Training loop ----
    best_reward_ema = -float("inf")
    reward_ema = None
    reward_ema_beta = 0.99
    kl_coef = args.kl_coef  # adaptive: will be adjusted each step if kl_target > 0

    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        for batch_idx, (style_pm1, content_pm1) in enumerate(dataloader):
            t0 = time.time()
            B = style_pm1.shape[0]

            style_dev   = style_pm1.to(DEV, non_blocking=True)
            content_dev = content_pm1.to(DEV, non_blocking=True)

            # ==============================================================
            # STEP 1: ROLLOUT
            # ==============================================================
            model.eval()
            if args.use_lora:
                set_lora_enabled(model, True)
            token_trajs, gen_images_01 = rollout_generate(
                model, vae, style_dev, content_dev,
                G=args.G, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
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

                log_ratio = new_lp.float() - old_lp.float()
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
                policy_loss = -torch.min(surr1, surr2).mean()

                # Non-negative KL estimator (Schulman): (ratio - 1) - log(ratio) >= 0
                log_ratio_kl = (new_lp.float() - ref_lp.float()).clamp(-10, 10)
                kl_loss = kl_coef * (torch.exp(log_ratio_kl) - 1 - log_ratio_kl).mean()

                loss = (policy_loss + kl_loss) / args.G

                # Safety: skip if loss is NaN/Inf to prevent weight corruption
                if torch.isfinite(loss):
                    loss.backward()
                    total_policy_loss += policy_loss.item() / args.G
                    total_kl_loss     += kl_loss.item() / args.G
                else:
                    print(f"  [WARN] Skipping NaN/Inf loss at g={g}", flush=True)
                    del new_lp, log_ratio, ratio, adv_exp, surr1, surr2
                    del log_ratio_kl, policy_loss, kl_loss, loss
                    optimizer.zero_grad()  # discard any accumulated grads
                    torch.cuda.empty_cache()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                train_params, max_norm=args.grad_clip)
            optimizer.step()

            # ---- Adaptive KL coefficient ----
            # Only start adapting after 50 steps (KL is naturally small early on)
            if args.kl_target > 0 and total_kl_loss > 0 and global_step >= 50:
                raw_kl = total_kl_loss / kl_coef  # un-weighted KL
                if raw_kl > 2.0 * args.kl_target:
                    kl_coef = min(kl_coef * 1.1, 0.5)   # gentle increase, cap at 0.5
                elif raw_kl < 0.5 * args.kl_target:
                    kl_coef = max(kl_coef / 1.1, 1e-4)   # gentle decrease, floor at 1e-4

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
                print(
                    f"[Ep {epoch+1}/{args.epochs}] "
                    f"Step {global_step:5d} | "
                    f"loss={total_policy_loss + total_kl_loss:.4f} "
                    f"policy={total_policy_loss:.4f} "
                    f"kl={total_kl_loss:.4f} kl_coef={kl_coef:.4f} | "
                    f"R={rewards_mean:.4f} R_ema={reward_ema:.4f} "
                    f"[c={r_content_mean:.4f} s={r_style_mean:.4f} ssim={r_ssim_mean:.4f} tv={r_tv_mean:.4f}] | "
                    f"grad={grad_norm:.3f} | "
                    f"mem={mem:.1f}G | "
                    f"{dt:.1f}s/step"
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "grpo/policy_loss": total_policy_loss,
                        "grpo/kl_loss": total_kl_loss,
                        "grpo/total_loss": total_policy_loss + total_kl_loss,
                        "grpo/reward_mean": rewards_mean,
                        "grpo/reward_ema": reward_ema,
                        "reward/content (neg LPIPS)": r_content_mean,
                        "reward/style (neg Gram)": r_style_mean,
                        "reward/ssim (structure)": r_ssim_mean,
                        "reward/tv (neg TV)": r_tv_mean,
                        "grpo/grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                        "grpo/kl_coef": kl_coef,
                        "grpo/lr": optimizer.param_groups[0]["lr"],
                        "grpo/mem_gb": mem,
                    }, step=global_step)

            # ---- Rolling checkpoint (5 slots) ----
            if global_step % args.save_every == 0:
                roll_idx = (global_step // args.save_every) % 5
                p = _save_grpo_ckpt(model, optimizer, scaler, global_step, epoch, args,
                                     f"roll{roll_idx}")
                print(f"[ckpt] {os.path.basename(p)} (step={global_step})")
                _save_grpo_ckpt(model, optimizer, scaler, global_step, epoch, args, "latest")
                if reward_ema > best_reward_ema:
                    best_reward_ema = reward_ema
                    _save_grpo_ckpt(model, optimizer, scaler, global_step, epoch, args, "best")
                    print(f"[ckpt] best updated (R_ema={reward_ema:.4f})")

        # ---- Epoch end ----
        ep_time = time.time() - t_epoch
        print(f"[Ep {epoch+1}] finished in {ep_time/60:.1f}min")
        _save_grpo_ckpt(model, optimizer, scaler, global_step, epoch + 1, args,
                         f"ep{epoch+1}")
        _save_grpo_ckpt(model, optimizer, scaler, global_step, epoch + 1, args, "latest")

    # ---- Final save ----
    p = _save_grpo_ckpt(model, optimizer, scaler, global_step, args.epochs, args, "final")
    print(f"[GRPO] Final checkpoint: {p}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
