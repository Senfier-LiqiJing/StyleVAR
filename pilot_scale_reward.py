"""Pilot Study 1: Scale-Reward Gradient Sensitivity Heatmap.

Question: For each reward (LPIPS / CLIP / DreamSim / StyleLoss / SSIM), which VAR
scales (1x1 -> 16x16) receive the strongest gradient signal?

Method (Option B, differentiable "soft rollout"):
  - Replace sampling with softmax(logits) @ codebook.embedding.weight
  - Replace in-place f_hat accumulation with non-in-place add
  - Backward each reward -> collect ||grad(logits_at_scale_t)||_2
  - Average over N batches, normalize per-reward, plot heatmap

Data:
  - Content: COCO train2017 (data/coco2017/images/train2017)
  - Style:   WikiArt (data/wikiart)
  - Decoupled content-style pairs (random pairing)

Usage (on the GPU machine with full data):
  CUDA_VISIBLE_DEVICES=0,1 python pilot_scale_reward.py \
      --sft_ckpt Output_v2/ar-ckpt-best.pth \
      --grpo_ckpt grpo_output_v3/grpo_best.pth \
      --content_dir data/coco2017/images/train2017 \
      --style_dir   data/wikiart \
      --num_batches 8 --batch_size 2 \
      --out_dir pilot_results

Output:
  pilot_results/scale_reward_heatmap.png
  pilot_results/scale_reward_raw.csv
  pilot_results/scale_reward_normalized.csv
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision.transforms import transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import build_vae_stylevar, VQVAE, StyleVAR


# =========================== Dataset =======================================
class UnpairedPairDataset(Dataset):
    EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp",
            "*.JPG", "*.JPEG", "*.PNG", "*.WEBP")

    def __init__(self, content_dir: str, style_dir: str, transform, n_cap: int = 20000):
        self.content_files, self.style_files = [], []
        for ext in self.EXTS:
            self.content_files.extend(glob.glob(os.path.join(content_dir, "**", ext), recursive=True))
            self.style_files.extend(glob.glob(os.path.join(style_dir, "**", ext), recursive=True))
        self.content_files.sort(); self.style_files.sort()
        self.content_files = self.content_files[:n_cap]
        self.style_files   = self.style_files[:n_cap]
        assert len(self.content_files) > 0, f"No content in {content_dir}"
        assert len(self.style_files)   > 0, f"No style in {style_dir}"
        self.transform = transform
        print(f"[Data] content={len(self.content_files)}  style={len(self.style_files)}")

    def __len__(self):
        return min(len(self.content_files), len(self.style_files))

    def __getitem__(self, idx):
        content = Image.open(self.content_files[idx]).convert("RGB")
        style   = Image.open(self.style_files[random.randint(0, len(self.style_files) - 1)]).convert("RGB")
        return self.transform(style), self.transform(content)


# =========================== LoRA (mirror of train_grpo) ===================
class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.base_weight  = base_linear.weight
        self.base_weight.requires_grad_(False)
        self.has_bias = base_linear.bias is not None
        if self.has_bias:
            self.base_bias = base_linear.bias
            self.base_bias.requires_grad_(False)
        dev, dtype = base_linear.weight.device, base_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self._enabled = True

    @property
    def weight(self):
        if self._enabled:
            return self.base_weight + (self.lora_B @ self.lora_A) * self.scaling
        return self.base_weight

    @property
    def bias(self):
        return self.base_bias if self.has_bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def apply_lora(model, rank, alpha):
    for block in model.blocks:
        for attr_name in ("mat_qkv_guide", "mat_qkv_target", "proj"):
            old = getattr(block.attn, attr_name)
            setattr(block.attn, attr_name, LoRALinear(old, rank, alpha))
        for attr_name in ("fc1", "fc2"):
            old = getattr(block.ffn, attr_name)
            setattr(block.ffn, attr_name, LoRALinear(old, rank, alpha))


# =========================== Soft (differentiable) rollout =================
@torch.enable_grad()
def soft_rollout(actor: StyleVAR, vae: VQVAE,
                 style_pm1: torch.Tensor, content_pm1: torch.Tensor,
                 temperature: float = 1.0
                 ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Differentiable rollout using softmax-weighted codebook.

    Returns:
      gen_01:          (B, 3, H, W) in [0,1], differentiable wrt per-scale logits
      logits_per_scale: list of (B, pn_t^2, V) tensors with retain_grad()
    """
    DEV = style_pm1.device
    B   = style_pm1.shape[0]
    patch_nums = actor.patch_nums
    quantize   = vae.quantize
    SN         = len(patch_nums)
    HW_last    = patch_nums[-1]

    # Inputs must carry requires_grad so downstream tensors (and thus logits) get
    # a valid autograd graph even when all model parameters are frozen.
    content_pm1 = content_pm1.detach().clone().requires_grad_(True)
    style_pm1   = style_pm1.detach().clone().requires_grad_(True)

    # ---- Conditioning (style/content encoders) ----
    sos     = actor.feat_emb(actor.content_encoder(content_pm1).squeeze(-1).squeeze(-1))
    cond_BD = actor.feat_emb(actor.style_encoder(style_pm1).squeeze(-1).squeeze(-1))

    lvl_pos = actor.lvl_embed(actor.lvl_1L) + actor.pos_1LC

    # Style / content multi-scale token features (for cross-attn in blocks)
    with torch.no_grad():
        ms_style_idx    = vae.img_to_idxBl(style_pm1)
        ms_style_BlCv   = quantize.msBllist_to_BlCv_list(ms_style_idx)
        ms_content_idx  = vae.img_to_idxBl(content_pm1)
        ms_content_BlCv = quantize.msBllist_to_BlCv_list(ms_content_idx)
    ms_style_BlC   = [actor.word_embed(item) for item in ms_style_BlCv]
    ms_content_BlC = [actor.word_embed(item) for item in ms_content_BlCv]

    cur_L = 0
    for idx_s, sBlC in enumerate(ms_style_BlC):
        pn = patch_nums[idx_s]
        ms_style_BlC[idx_s]   = sBlC + lvl_pos[:, cur_L:cur_L + pn**2]
        ms_content_BlC[idx_s] = ms_content_BlC[idx_s] + lvl_pos[:, cur_L:cur_L + pn**2]
        cur_L += pn**2

    # ---- Init next_token_map from SOS ----
    next_token_map = (
        sos.unsqueeze(1).expand(B, actor.first_l, -1)
        + actor.pos_start.expand(B, actor.first_l, -1)
        + lvl_pos[:, :actor.first_l]
    )

    # f_hat accumulator (NON-in-place for autograd)
    f_hat = sos.new_zeros(B, actor.Cvae, HW_last, HW_last)

    logits_per_scale: List[torch.Tensor] = []

    # KV caching disabled for cleanliness (each scale runs its own forward)
    for b in actor.blocks:
        b.attn.kv_caching(False)

    cur_L = 0
    for si, pn in enumerate(patch_nums):
        cur_L += pn * pn
        cond_BD_or_gss = actor.shared_ada_lin(cond_BD)
        x = next_token_map
        for b in actor.blocks:
            x = b(x=x, style=ms_style_BlC[si], content=ms_content_BlC[si],
                  cond_BD=cond_BD_or_gss, attn_bias=None,
                  alpha=actor.alpha_nums[si])

        logits_BlV = actor.get_logits(x, cond_BD)           # (B, pn^2, V)
        if temperature != 1.0:
            logits_BlV = logits_BlV / temperature
        # Keep gradient for gradient-sensitivity analysis
        logits_BlV = logits_BlV.float()
        logits_BlV.retain_grad()
        logits_per_scale.append(logits_BlV)

        # Soft embedding: probs (B, pn^2, V) @ codebook (V, Cvae) = (B, pn^2, Cvae)
        probs    = F.softmax(logits_BlV, dim=-1)
        codebook = quantize.embedding.weight.to(probs.dtype)
        h_BlCvae = probs @ codebook                                   # (B, pn^2, Cvae)
        h_BChw   = h_BlCvae.transpose(1, 2).reshape(B, actor.Cvae, pn, pn)

        # Non-in-place f_hat update (mirrors Phi-based quant_resi logic)
        ratio = si / max(1, SN - 1)
        if si != SN - 1:
            h_up = F.interpolate(h_BChw, size=(HW_last, HW_last), mode='bicubic')
            h_refined = quantize.quant_resi[ratio](h_up)
            f_hat = f_hat + h_refined
            next_feat = F.interpolate(f_hat, size=(patch_nums[si+1], patch_nums[si+1]), mode='area')
        else:
            h_refined = quantize.quant_resi[ratio](h_BChw)
            f_hat = f_hat + h_refined
            next_feat = f_hat   # unused but keep symmetry

        if si != actor.num_stages_minus_1:
            next_pn  = patch_nums[si + 1]
            next_flat = next_feat.view(B, actor.Cvae, -1).transpose(1, 2)
            next_token_map = (
                actor.word_embed(next_flat)
                + lvl_pos[:, cur_L:cur_L + next_pn**2]
            )

    # Decode to image (non-in-place: avoid .clamp_)
    img_pm1 = vae.decoder(vae.post_quant_conv(f_hat))
    img_pm1 = torch.clamp(img_pm1, -1.0, 1.0)
    gen_01  = (img_pm1 + 1.0) * 0.5
    return gen_01, logits_per_scale


# =========================== Rewards (all differentiable) ==================
class LPIPSWrap(nn.Module):
    def __init__(self, device):
        super().__init__()
        import lpips
        self.net = lpips.LPIPS(net="alex").to(device).eval()
        for p in self.parameters(): p.requires_grad_(False)

    def forward(self, gen_01, ref_01):
        gen_pm1 = gen_01 * 2 - 1
        ref_pm1 = ref_01 * 2 - 1
        return -self.net(gen_pm1, ref_pm1).view(-1)  # higher = better


class CLIPWrap(nn.Module):
    """Image-image cosine similarity using CLIP ViT-B/32.

    Loads in this priority order (to avoid open_clip's blocked OpenAI CDN):
      1. --clip_local_dir  (offline, e.g. a snapshot of openai/clip-vit-base-patch32)
      2. HuggingFace transformers CLIPModel  (respects HF_ENDPOINT mirror env var)
      3. open_clip  (last resort — usually blocked in mainland China)
    """
    def __init__(self, device, local_dir: str = ""):
        super().__init__()
        self.backend = None
        # Try local HF snapshot first
        if local_dir and os.path.isdir(local_dir):
            from transformers import CLIPModel
            self.model = CLIPModel.from_pretrained(local_dir).to(device).eval()
            self.backend = f"hf-local:{local_dir}"
        # Then HF hub (uses HF_ENDPOINT=https://hf-mirror.com if set)
        if self.backend is None:
            try:
                from transformers import CLIPModel
                self.model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32").to(device).eval()
                self.backend = "hf-hub"
            except Exception as e:
                print(f"[CLIP] HF hub failed: {e}")
        # Last resort: open_clip
        if self.backend is None:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai")
            self.model = model.to(device).eval()
            self.backend = "open_clip"
        print(f"[CLIP] loaded via backend={self.backend}")
        for p in self.parameters(): p.requires_grad_(False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean); self.register_buffer("std", std)

    def _encode(self, img_01):
        x = F.interpolate(img_01, size=224, mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        if self.backend == "open_clip":
            feat = self.model.encode_image(x)
        else:
            out = self.model.get_image_features(pixel_values=x)
            if isinstance(out, torch.Tensor):
                feat = out
            elif hasattr(out, "image_embeds"):
                feat = out.image_embeds
            elif hasattr(out, "pooler_output"):
                # Some transformers versions return vision-model output here.
                feat = self.model.visual_projection(out.pooler_output)
            else:
                raise RuntimeError(f"Unexpected CLIP output type: {type(out)}")
        return F.normalize(feat.float(), dim=-1)

    def forward(self, gen_01, ref_01):
        fg = self._encode(gen_01); fr = self._encode(ref_01)
        return (fg * fr).sum(dim=-1)  # cosine in [-1,1], higher = more similar


class DreamSimWrap(nn.Module):
    def __init__(self, device):
        super().__init__()
        from dreamsim import dreamsim
        import sys
        saved_utils = {}
        for k in list(sys.modules.keys()):
            if k == "utils" or k.startswith("utils."):
                saved_utils[k] = sys.modules.pop(k)
        cache_dir = os.path.join(ROOT, "ckpt", "dreamsim")
        try:
            self.model, _ = dreamsim(pretrained=True, device=device,
                                     dreamsim_type="dino_vitb16", cache_dir=cache_dir)
        finally:
            for k in list(sys.modules.keys()):
                if k == "utils" or k == "vision_transformer" or k.startswith("utils."):
                    sys.modules.pop(k, None)
            sys.modules.update(saved_utils)
        for p in self.parameters(): p.requires_grad_(False)

    def forward(self, gen_01, ref_01):
        if gen_01.shape[-1] != 224:
            gen_01 = F.interpolate(gen_01, 224, mode='bilinear', align_corners=False)
        if ref_01.shape[-1] != 224:
            ref_01 = F.interpolate(ref_01, 224, mode='bilinear', align_corners=False)
        return -self.model(gen_01, ref_01).view(-1)


class VGGGramWrap(nn.Module):
    """Negative VGG-19 Gram-matrix MSE. Differentiable (no torch.no_grad)."""
    STYLE_LAYERS = ["0", "5", "10", "19", "28"]
    def __init__(self, device):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        prev = 0
        for idx_str in self.STYLE_LAYERS:
            idx = int(idx_str) + 1
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:idx]))
            prev = idx
        for p in self.parameters(): p.requires_grad_(False)
        self.eval().to(device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _gram(x):
        B, C, H, W = x.shape
        f = x.view(B, C, H * W)
        return torch.bmm(f, f.transpose(1, 2)) / (C * H * W)

    def forward(self, gen_01, ref_01):
        gen = (gen_01 - self.mean) / self.std
        ref = (ref_01 - self.mean) / self.std
        loss = gen.new_zeros(gen.shape[0])
        xg, xr = gen, ref
        for sl in self.slices:
            xg = sl(xg); xr = sl(xr)
            loss = loss + (self._gram(xg) - self._gram(xr)).square().mean(dim=(1, 2))
        return -loss  # higher = more similar style


class SSIMWrap(nn.Module):
    def __init__(self, device, window_size=11):
        super().__init__()
        self.C1, self.C2 = 0.01**2, 0.03**2
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
        w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0).expand(3, 1, window_size, window_size)
        self.register_buffer("window", w)
        self.pad = window_size // 2

    def forward(self, gen_01, ref_01):
        w, pad = self.window.to(gen_01.device), self.pad
        mu_x = F.conv2d(gen_01, w, padding=pad, groups=3)
        mu_y = F.conv2d(ref_01, w, padding=pad, groups=3)
        mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
        sx = F.conv2d(gen_01.square(), w, padding=pad, groups=3) - mu_x2
        sy = F.conv2d(ref_01.square(), w, padding=pad, groups=3) - mu_y2
        sxy = F.conv2d(gen_01 * ref_01, w, padding=pad, groups=3) - mu_xy
        num = (2 * mu_xy + self.C1) * (2 * sxy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sx + sy + self.C2)
        return (num / den).mean(dim=(1, 2, 3))  # per-sample SSIM in [0,1]


# =========================== Model loading =================================
def load_model(args, device):
    patch_nums = tuple(int(x) for x in "1_2_3_4_5_6_8_10_13_16".split("_"))
    vae, model = build_vae_stylevar(
        device=device, patch_nums=patch_nums,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        depth=20, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
        style_enc_dim=512,
    )
    vae.load_state_dict(torch.load(os.path.join(ROOT, args.vae_ckpt), map_location="cpu"), strict=True)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)

    sft_ckpt = torch.load(os.path.join(ROOT, args.sft_ckpt), map_location="cpu")
    if "trainer" in sft_ckpt and "var_wo_ddp" in sft_ckpt["trainer"]:
        sft_state = sft_ckpt["trainer"]["var_wo_ddp"]
    elif "model" in sft_ckpt:
        sft_state = sft_ckpt["model"]
    else:
        sft_state = sft_ckpt
    model.load_state_dict(sft_state, strict=True)
    print(f"[SFT] loaded: {args.sft_ckpt}")

    if args.grpo_ckpt:
        gckpt = torch.load(os.path.join(ROOT, args.grpo_ckpt), map_location="cpu")
        gargs = gckpt.get("args", {})
        rank  = gargs.get("lora_rank", 256)
        alpha = gargs.get("lora_alpha", 512.0)
        apply_lora(model, rank, alpha)
        sd = model.state_dict(); sd.update(gckpt["model"]); model.load_state_dict(sd)
        print(f"[GRPO] LoRA loaded (rank={rank}): {args.grpo_ckpt}")

    model.eval()
    # Keep only learnable params (for LoRA case) requires_grad True;
    # Soft rollout gradients actually back-prop to the retained logits, not model params.
    # We only need logits.grad, so set everything frozen.
    for p in model.parameters(): p.requires_grad_(False)
    return vae, model, patch_nums


# =========================== Pilot core ====================================
def run_pilot(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae, model, patch_nums = load_model(args, device)

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = UnpairedPairDataset(args.content_dir, args.style_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=True)

    # Rewards (each against reference image: content for LPIPS/CLIP/DreamSim/SSIM, style for VGG-Gram)
    rewards = {}
    reward_ref = {}  # 'content' or 'style'
    print("[Rewards] building...")
    rewards["LPIPS"]     = LPIPSWrap(device).to(device);     reward_ref["LPIPS"]     = "content"
    try:
        rewards["CLIP"]  = CLIPWrap(device, local_dir=args.clip_local_dir).to(device)
        reward_ref["CLIP"] = "content"
    except Exception as e:
        print(f"[Rewards] CLIP unavailable: {e}")
    try:
        rewards["DreamSim"] = DreamSimWrap(device).to(device);reward_ref["DreamSim"] = "content"
    except Exception as e:
        print(f"[Rewards] DreamSim unavailable: {e}")
    rewards["StyleLoss"] = VGGGramWrap(device).to(device);   reward_ref["StyleLoss"] = "style"
    rewards["SSIM"]      = SSIMWrap(device).to(device);      reward_ref["SSIM"]      = "content"
    for r in rewards.values():
        for p in r.parameters(): p.requires_grad_(False)

    reward_names = list(rewards.keys())
    SN = len(patch_nums)

    # Accumulator: sum of grad_l2 over batches, [reward, scale]
    grad_sum = np.zeros((len(reward_names), SN), dtype=np.float64)
    grad_cnt = 0

    t0 = time.time()
    batches_done = 0
    for batch_idx, batch in enumerate(loader):
        if batches_done >= args.num_batches: break
        style_pm1, content_pm1 = batch
        style_pm1   = style_pm1.to(device, non_blocking=True)
        content_pm1 = content_pm1.to(device, non_blocking=True)
        content_01  = (content_pm1 + 1) * 0.5
        style_01    = (style_pm1   + 1) * 0.5

        # --- Soft rollout (shared across rewards to avoid re-compute) ---
        # But autograd needs a fresh graph per backward, so we rollout once per reward.
        for r_idx, rname in enumerate(reward_names):
            gen_01, logits_list = soft_rollout(model, vae, style_pm1, content_pm1)
            ref_img = content_01 if reward_ref[rname] == "content" else style_01
            r_vals  = rewards[rname](gen_01, ref_img)     # (B,)
            loss    = -r_vals.mean()                       # maximize reward = minimize -reward

            # Compute gradient w.r.t. all scale logits in one backward
            grads = torch.autograd.grad(loss, logits_list, retain_graph=False, allow_unused=False)
            # Per-scale grad L2 norm (mean over batch)
            for si, g in enumerate(grads):
                # g shape (B, pn^2, V); normalize by sqrt(num_tokens * V) so scales are comparable?
                # We report RAW L2 first, then also per-token-mean for fairness.
                gn = g.detach().float().pow(2).sum(dim=(1, 2)).sqrt().mean().item()  # mean batch L2 norm
                grad_sum[r_idx, si] += gn

            # Free memory
            del gen_01, logits_list, r_vals, loss, grads
            torch.cuda.empty_cache()

        grad_cnt += 1
        batches_done += 1
        print(f"[batch {batches_done}/{args.num_batches}] dt={time.time()-t0:.1f}s", flush=True)

    assert grad_cnt > 0, "No batches processed"
    grad_mean = grad_sum / grad_cnt   # [R, SN]

    # Row-normalized version (each reward sums to 1 across scales)
    grad_norm = grad_mean / (grad_mean.sum(axis=1, keepdims=True) + 1e-12)
    # Also per-token normalized (divide by num tokens at that scale)
    tokens_per_scale = np.array([pn * pn for pn in patch_nums], dtype=np.float64)
    grad_pertoken = grad_mean / tokens_per_scale[None, :]
    grad_pertoken_norm = grad_pertoken / (grad_pertoken.sum(axis=1, keepdims=True) + 1e-12)

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    _save_csv(os.path.join(args.out_dir, "scale_reward_raw.csv"),       reward_names, patch_nums, grad_mean)
    _save_csv(os.path.join(args.out_dir, "scale_reward_rownorm.csv"),   reward_names, patch_nums, grad_norm)
    _save_csv(os.path.join(args.out_dir, "scale_reward_pertoken.csv"),  reward_names, patch_nums, grad_pertoken)
    _save_csv(os.path.join(args.out_dir, "scale_reward_pertoken_rownorm.csv"),
              reward_names, patch_nums, grad_pertoken_norm)

    _plot_heatmap(os.path.join(args.out_dir, "scale_reward_heatmap.png"),
                  reward_names, patch_nums, grad_norm,
                  title="Row-normalized ||grad_logits|| per (reward, scale)")
    _plot_heatmap(os.path.join(args.out_dir, "scale_reward_heatmap_pertoken.png"),
                  reward_names, patch_nums, grad_pertoken_norm,
                  title="Per-token normalized ||grad_logits|| per (reward, scale)")
    print(f"[Done] results in {args.out_dir}")


def _save_csv(path, reward_names, patch_nums, mat):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reward"] + [f"{pn}x{pn}" for pn in patch_nums])
        for rname, row in zip(reward_names, mat):
            w.writerow([rname] + [f"{v:.6e}" for v in row])


def _plot_heatmap(path, reward_names, patch_nums, mat, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.1 * len(patch_nums) + 2, 0.7 * len(reward_names) + 2))
    im = ax.imshow(mat, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(patch_nums))); ax.set_xticklabels([f"{pn}²" for pn in patch_nums])
    ax.set_yticks(range(len(reward_names))); ax.set_yticklabels(reward_names)
    ax.set_xlabel("VAR scale"); ax.set_ylabel("Reward"); ax.set_title(title)
    # Annotate values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center',
                    color='white' if mat[i,j] < mat.max()*0.6 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================ CLI ==========================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_ckpt",   type=str, default="Output_v2/ar-ckpt-best.pth")
    p.add_argument("--grpo_ckpt",  type=str, default="",
                   help="Optional GRPO LoRA ckpt; empty = analyze SFT only")
    p.add_argument("--vae_ckpt",   type=str, default="ckpt/vae_ch160v4096z32.pth")
    p.add_argument("--content_dir", type=str, default="data/coco2017/images/train2017")
    p.add_argument("--style_dir",   type=str, default="data/wikiart")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_batches", type=int, default=8)
    p.add_argument("--out_dir",    type=str, default="pilot_results")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--clip_local_dir", type=str, default="",
                   help="Local HF snapshot of openai/clip-vit-base-patch32 "
                        "(recommended in China to skip blocked OpenAI CDN)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    run_pilot(args)


if __name__ == "__main__":
    main()
