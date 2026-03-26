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
    p = argparse.ArgumentParser("GRPO training for StyleVAR (LoRA)")
    # paths
    p.add_argument("--content_dir", type=str, default="data/coco2017",
                    help="Directory with content images (recursive search)")
    p.add_argument("--style_dir",   type=str, default="data/wikiart",
                    help="Directory with style images (recursive search)")
    p.add_argument("--vae_ckpt",  type=str, default="ckpt/vae_ch160v4096z32.pth")
    p.add_argument("--var_ckpt",  type=str, default="ckpt/style_var_d20_11_20_21.pth")
    p.add_argument("--out_dir",   type=str, default="./grpo_outputs")
    # GRPO
    p.add_argument("--G",            type=int,   default=4,     help="Group size (rollouts per prompt)")
    p.add_argument("--kl_coef",      type=float, default=0.04,  help="KL penalty coefficient beta")
    p.add_argument("--clip_eps",     type=float, default=0.2,   help="PPO-style clipping epsilon")
    # reward weights
    p.add_argument("--lam_content",  type=float, default=1.0,   help="Weight for LPIPS content reward")
    p.add_argument("--lam_style",    type=float, default=1.0,   help="Weight for VGG Gram style reward")
    p.add_argument("--lam_tv",       type=float, default=0.01,  help="Weight for TV quality reward")
    # training
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=2,     help="Condition pairs per step")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--save_every",   type=int,   default=200)
    p.add_argument("--log_every",    type=int,   default=10)
    # sampling
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--top_k",       type=int,   default=900)
    p.add_argument("--top_p",       type=float, default=0.96)
    # model
    p.add_argument("--depth",       type=int,   default=20)
    # LoRA
    p.add_argument("--lora_rank",   type=int,   default=16,    help="LoRA rank")
    p.add_argument("--lora_alpha",  type=float, default=32.0,  help="LoRA alpha (scaling = alpha/rank)")
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
    """Freeze entire model, then inject LoRA into every attention layer.

    Target layers per block:
      - mat_qkv_guide  (C -> 3C, bias=False)  — Q/K/V for guide stream
      - mat_qkv_target (C -> 3C, bias=False)  — Q/K/V for target stream
      - proj           (C -> C,  bias=True)   — output projection

    Returns the list of trainable LoRA parameters.
    """
    # 1. Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    lora_params: List[nn.Parameter] = []
    for block in model.blocks:
        attn = block.attn
        for attr_name in ("mat_qkv_guide", "mat_qkv_target", "proj"):
            old_linear = getattr(attn, attr_name)
            new_module = LoRALinear(old_linear, rank, alpha)
            setattr(attn, attr_name, new_module)
            lora_params.extend([new_module.lora_A, new_module.lora_B])

    total = sum(p.numel() for p in lora_params)
    print(f"[LoRA] Injected into {len(model.blocks)} blocks, "
          f"{len(lora_params)} adapter tensors, {total:,} trainable params")
    return lora_params


def set_lora_enabled(model: StyleVAR, enabled: bool):
    """Toggle LoRA on/off.  Off = reference policy (base model only)."""
    for block in model.blocks:
        attn = block.attn
        for attr_name in ("mat_qkv_guide", "mat_qkv_target", "proj"):
            module = getattr(attn, attr_name)
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
    Build a SINGLE StyleVAR + VAE on cuda:0, apply LoRA.
    No duplicate reference model needed (LoRA toggle serves as reference).
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

    # Load VAE weights (frozen via test_mode=True in VQVAE)
    vae_state = torch.load(args.vae_ckpt, map_location="cpu")
    vae.load_state_dict(vae_state, strict=True)
    print(f"[GRPO] VAE loaded from {args.vae_ckpt}")

    # Load StyleVAR checkpoint
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

    # Inject LoRA (freezes base, returns trainable params)
    lora_params = apply_lora(model, args.lora_rank, args.lora_alpha)

    return model, vae, lora_params, patch_nums


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
    lam_content: float,
    lam_style: float,
    lam_tv: float,
) -> torch.Tensor:
    """Compute composite reward R_i for each generated image.  Returns (B*G,)."""
    BG = gen_images_01.shape[0]

    # Expand conditions to match (B*G, ...)
    style_exp   = style_img_01.repeat_interleave(G, dim=0)
    content_exp = content_img_01.repeat_interleave(G, dim=0)

    # Content reward: negative LPIPS (expects [-1,1])
    gen_pm1     = gen_images_01 * 2 - 1
    content_pm1 = content_exp * 2 - 1
    r_content   = -lpips_net(gen_pm1, content_pm1).view(BG)

    # Style reward: negative VGG Gram MSE (expects [0,1])
    r_style = vgg_gram(gen_images_01, style_exp)

    # Quality reward: negative TV
    r_tv = tv_reward(gen_images_01)

    return lam_content * r_content + lam_style * r_style + lam_tv * r_tv


def compute_group_advantages(rewards: torch.Tensor, G: int) -> torch.Tensor:
    """Group-relative advantage: A_i = (R_i - mu) / (sigma + eps)."""
    B = rewards.shape[0] // G
    R = rewards.view(B, G)
    mu    = R.mean(dim=1, keepdim=True)
    sigma = R.std(dim=1, keepdim=True)
    A = (R - mu) / (sigma + 1e-4)
    return A.view(B * G)


# ======================== Main Training Loop ===============================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    DEV0 = torch.device("cuda:0")
    DEV1 = torch.device("cuda:1")

    print("=" * 60)
    print("  GRPO Training for StyleVAR  (LoRA)")
    print(f"  Model + VAE on {DEV0},  Rewards on {DEV1}")
    print(f"  G={args.G}, kl_coef={args.kl_coef}, clip_eps={args.clip_eps}")
    print(f"  LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    print("=" * 60)

    # ---- 1. Build model with LoRA ----
    model, vae, lora_params, patch_nums = _build_model(args)
    L = sum(pn ** 2 for pn in patch_nums)  # 343

    # ---- 2. Reward models on cuda:1 ----
    lpips_net = _load_lpips_alex(DEV1)
    vgg_gram  = VGGGramStyleReward().to(DEV1)
    tv_reward = TVReward().to(DEV1)
    print("[GRPO] Reward models loaded on cuda:1")

    # ---- 3. Optimizer (LoRA params only) ----
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr,
                                   betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()   # mixed precision

    # ---- 4. Dataset (unpaired content + style) ----
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        normalize_01_into_pm1,              # -> [-1, 1]
    ])
    dataset = UnpairedStyleContentDataset(
        content_dir=args.content_dir, style_dir=args.style_dir,
        transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    print(f"[GRPO] Dataset: {len(dataset)} samples, bs={args.batch_size}")

    # ---- 5. Training loop ----
    global_step = 0
    for epoch in range(args.epochs):
        t_epoch = time.time()
        for batch_idx, (style_pm1, content_pm1) in enumerate(dataloader):
            t0 = time.time()
            B = style_pm1.shape[0]

            style_dev0   = style_pm1.to(DEV0, non_blocking=True)
            content_dev0 = content_pm1.to(DEV0, non_blocking=True)

            # ==============================================================
            # STEP 1: ROLLOUT — generate G variations per (content, style)
            # ==============================================================
            model.eval()
            set_lora_enabled(model, True)       # sample from current policy
            token_trajs, gen_images_01 = rollout_generate(
                model, vae, style_dev0, content_dev0,
                G=args.G, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
            )

            # ==============================================================
            # STEP 2: OLD LOG-PROBS (current policy, no grad)
            # ==============================================================
            with torch.no_grad():
                style_feat, content_feat = precompute_style_content_features(
                    vae, style_dev0, content_dev0)
                old_logprobs_list = []
                for g in range(args.G):
                    lp = compute_logprobs_single(
                        model, vae, token_trajs[g],
                        style_dev0, content_dev0,
                        style_feat, content_feat)
                    old_logprobs_list.append(lp)
                old_logprobs = torch.cat(old_logprobs_list, dim=0)  # (B*G, L)

            # ==============================================================
            # STEP 3: REFERENCE LOG-PROBS (base model = LoRA disabled)
            # ==============================================================
            with torch.no_grad():
                set_lora_enabled(model, False)
                ref_logprobs_list = []
                for g in range(args.G):
                    lp = compute_logprobs_single(
                        model, vae, token_trajs[g],
                        style_dev0, content_dev0,
                        style_feat, content_feat)
                    ref_logprobs_list.append(lp)
                ref_logprobs = torch.cat(ref_logprobs_list, dim=0)  # (B*G, L)
                set_lora_enabled(model, True)

            # ==============================================================
            # STEP 4: REWARDS on cuda:1
            # ==============================================================
            gen_dev1     = gen_images_01.to(DEV1)
            style_01     = style_dev0.add(1).mul(0.5).to(DEV1)
            content_01   = content_dev0.add(1).mul(0.5).to(DEV1)

            rewards = compute_rewards(
                gen_dev1, style_01, content_01, G=args.G,
                lpips_net=lpips_net, vgg_gram=vgg_gram, tv_reward=tv_reward,
                lam_content=args.lam_content, lam_style=args.lam_style,
                lam_tv=args.lam_tv,
            ).to(DEV0)                                         # (B*G,)
            advantages = compute_group_advantages(rewards, args.G)  # (B*G,)

            rewards_mean = rewards.mean().item()

            # Free reward-side memory
            del gen_dev1, style_01, content_01, gen_images_01
            torch.cuda.empty_cache()

            # ==============================================================
            # STEP 5: POLICY UPDATE (gradient accumulation across G)
            # ==============================================================
            model.train()
            optimizer.zero_grad()

            total_policy_loss = 0.0
            total_kl_loss = 0.0

            for g in range(args.G):
                # Forward with gradient (mixed precision)
                with torch.cuda.amp.autocast():
                    new_lp = compute_logprobs_single(
                        model, vae, token_trajs[g],
                        style_dev0, content_dev0,
                        style_feat, content_feat)          # (B, L)

                old_lp  = old_logprobs[g*B:(g+1)*B].detach()   # (B, L)
                ref_lp  = ref_logprobs[g*B:(g+1)*B].detach()   # (B, L)
                adv     = advantages[g*B:(g+1)*B].detach()      # (B,)

                # Per-token ratio & clipped surrogate
                log_ratio = new_lp.float() - old_lp.float()
                ratio = torch.exp(log_ratio)
                adv_exp = adv.unsqueeze(1).expand_as(ratio)

                surr1 = ratio * adv_exp
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps,
                                    1.0 + args.clip_eps) * adv_exp
                policy_loss = -torch.min(surr1, surr2).mean()

                # KL penalty
                kl_loss = args.kl_coef * (new_lp.float() - ref_lp.float()).mean()

                loss = (policy_loss + kl_loss) / args.G   # average over G
                scaler.scale(loss).backward()

                total_policy_loss += policy_loss.item() / args.G
                total_kl_loss     += kl_loss.item() / args.G

            # Gradient clipping & step
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                lora_params, max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # ---- Cleanup ----
            del old_logprobs, ref_logprobs, advantages, rewards
            del style_feat, content_feat, token_trajs
            gc.collect()
            torch.cuda.empty_cache()

            global_step += 1
            dt = time.time() - t0

            # ---- Logging ----
            if global_step % args.log_every == 0:
                mem0 = torch.cuda.max_memory_allocated(DEV0) / 1e9
                mem1 = torch.cuda.max_memory_allocated(DEV1) / 1e9
                print(
                    f"[Ep {epoch+1}/{args.epochs}] "
                    f"Step {global_step:5d} | "
                    f"loss={total_policy_loss + total_kl_loss:.4f} "
                    f"policy={total_policy_loss:.4f} "
                    f"kl={total_kl_loss:.4f} | "
                    f"R_mean={rewards_mean:.4f} | "
                    f"grad={grad_norm:.3f} | "
                    f"mem0={mem0:.1f}G mem1={mem1:.1f}G | "
                    f"{dt:.1f}s/step"
                )

            # ---- Checkpoint (LoRA weights only) ----
            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(
                    args.out_dir, f"grpo_lora_step{global_step}.pth")
                lora_state = {k: v for k, v in model.state_dict().items()
                              if "lora_" in k}
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "lora": lora_state,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                print(f"[GRPO] Saved LoRA checkpoint: {ckpt_path}")

        print(f"[Ep {epoch+1}] finished in {time.time()-t_epoch:.0f}s")

    # Final save
    ckpt_path = os.path.join(args.out_dir, "grpo_lora_final.pth")
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save({
        "step": global_step,
        "lora": lora_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }, ckpt_path)
    print(f"[GRPO] Final LoRA checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
