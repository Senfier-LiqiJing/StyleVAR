"""Evaluate a StyleVAR / GRPO checkpoint on 3 datasets:

  - OmniStyle-150k  (paired, in-distribution)
  - ImagePulse      (paired, in-distribution)
  - COCO + WikiArt  (unpaired cross, OOD)

Metrics:
  Paired (has target/GT):
    - Style Loss  (VGG19 Gram, gen vs style)        — lower better
    - Content Loss (VGG19 conv4_2, gen vs content)  — lower better
    - LPIPS (gen vs GT)                             — lower better
    - SSIM (gen vs GT)                              — higher better
    - DreamSim (gen vs GT)                          — lower better
    - CLIP-I similarity (gen vs GT)                 — higher better

  Unpaired (no GT, COCO+WikiArt):
    Same metrics but vs content (for LPIPS/SSIM/DreamSim/CLIP) and vs style (for Style Loss).

Usage:
  python eval_grpo.py \
      --ckpt ckpt/grpo-best.pth \
      --omnistyle_n 50 --imagepulse_n 50 --cocowiki_n 50 \
      --out_dir eval_out

  # Evaluate SFT baseline for comparison:
  python eval_grpo.py --ckpt ckpt/sft-best.pth --out_dir eval_out_sft

The script uses `_load_var_state_strict` (the safe loader) so it handles plain SFT,
LoRA-wrapped full_state, and already-baked GRPO ckpts.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models as tv_models
from torchvision import transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import build_vae_stylevar
from utils.lora import (
    bake_lora_into_plain as _bake_lora_into_plain,
    apply_lora_to_plain_state as _apply_lora_to_plain_state,
    _extract_state_dict,
)


# =========================== AdaIN baseline ===============================
class AdaINBaseline(nn.Module):
    """AdaIN style transfer baseline (Huang & Belongie, 2017).
    Encoder = VGG19 up to relu4_1; Decoder = mirror architecture (pre-trained).

    Weights `decoder.pth` from https://github.com/naoto0804/pytorch-AdaIN
    (see bash download_adain.sh).
    """
    def __init__(self, device, decoder_path: str):
        super().__init__()
        self.device = device
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
        self.encoder = nn.Sequential(*list(vgg.children())[:21]).to(device).eval()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(512, 256, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(256, 128, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(128, 128, 3), nn.ReLU(),
            nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(128, 64, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(64, 3, 3),
        ).to(device)
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        self.decoder.eval()
        for p in self.parameters(): p.requires_grad_(False)

    @staticmethod
    def _mean_std(feat, eps=1e-5):
        N, C = feat.shape[:2]
        var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
        std = var.sqrt().view(N, C, 1, 1)
        mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return mean, std

    def _adain(self, c_feat, s_feat):
        sm, ss = self._mean_std(s_feat)
        cm, cs = self._mean_std(c_feat)
        return (c_feat - cm) / cs * ss + sm

    @torch.no_grad()
    def infer(self, content_01, style_01, alpha: float = 1.0):
        """content / style: (B, 3, H, W) in [0, 1]. Returns gen in [0, 1]."""
        c = content_01.to(self.device); s = style_01.to(self.device)
        c_feat = self.encoder(c); s_feat = self.encoder(s)
        t = self._adain(c_feat, s_feat)
        t = alpha * t + (1 - alpha) * c_feat
        return self.decoder(t).clamp(0, 1)


# =========================== Model loading ================================
# LoRA / ckpt-format helpers are imported from utils.lora.


def load_model(ckpt_path: str, vae_ckpt: str, device: torch.device,
               base_ckpt_path: str = ""):
    patch_nums = tuple(int(x) for x in "1_2_3_4_5_6_8_10_13_16".split("_"))
    vae, model = build_vae_stylevar(
        device=device, patch_nums=patch_nums,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        depth=20, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
        style_enc_dim=512,
    )
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)

    raw = torch.load(ckpt_path, map_location="cpu")
    st = _extract_state_dict(raw)

    # Detect checkpoint type from its keys
    has_lora   = any(k.endswith(".lora_A") for k in st.keys())
    has_base   = any(k.endswith(".base_weight") for k in st.keys())   # LoRA-wrapped full_state
    has_plain  = any(k.endswith(".attn.mat_qkv_guide.weight") for k in st.keys())  # plain SFT/merged

    args_dict = raw.get("args", {}) if isinstance(raw, dict) else {}
    rank  = args_dict.get("lora_rank", 256)
    alpha = args_dict.get("lora_alpha", 512.0)

    if has_lora and not has_base and not has_plain:
        # --- Case 3: LoRA-only ckpt. Need a base to stack onto. ---
        if not base_ckpt_path:
            raise RuntimeError(
                f"[eval] {ckpt_path} is LoRA-only (no base_weight / weight keys).\n"
                f"       Supply --base_ckpt (e.g. ckpt/sft-best.pth) to stack the LoRA onto."
            )
        print(f"[eval] detected LoRA-only ckpt (rank={rank}, alpha={alpha})")
        print(f"[eval] loading base from {base_ckpt_path}")
        base_raw = torch.load(base_ckpt_path, map_location="cpu")
        base_state = _extract_state_dict(base_raw)
        # If base is itself LoRA-wrapped, bake it first
        if any(k.endswith(".base_weight") for k in base_state.keys()):
            b_args = base_raw.get("args", {}) if isinstance(base_raw, dict) else {}
            b_rank  = b_args.get("lora_rank", 256)
            b_alpha = b_args.get("lora_alpha", 512.0)
            print(f"[eval] base is LoRA-wrapped; baking (rank={b_rank}, alpha={b_alpha})")
            base_state = _bake_lora_into_plain(base_state, b_rank, b_alpha)
        final_state, n_applied = _apply_lora_to_plain_state(base_state, st, rank, alpha)
        print(f"[eval] applied {n_applied} LoRA modules onto base")
        st = final_state
    elif has_base:
        # --- Case 2: LoRA-wrapped full_state. Bake and load. ---
        print(f"[eval] detected LoRA-wrapped full_state ckpt; baking (rank={rank}, alpha={alpha})")
        st = _bake_lora_into_plain(st, rank, alpha)
    else:
        # --- Case 1: Plain ckpt. Direct strict load. ---
        print(f"[eval] detected plain ckpt")

    model.load_state_dict(st, strict=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    print(f"[eval] loaded {ckpt_path}  ({len(st)} tensors, strict=True)")
    return model, vae


# =========================== Metrics ========================================
class VGGStyleContentMetric(nn.Module):
    """VGG19 Gram + feature MSE. Same layers as the training reward."""
    CONTENT_LAYER = "21"  # conv4_2
    STYLE_LAYERS  = ["0", "5", "10", "19", "28"]
    def __init__(self, device):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.to(device).eval()
        for p in self.vgg.parameters(): p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)

    def _norm(self, x): return (x - self.mean) / self.std

    @staticmethod
    def _gram(f):
        b, c, h, w = f.shape
        feat = f.view(b, c, h * w)
        return torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)

    @torch.no_grad()
    def forward(self, gen_01, content_01, style_01):
        """gen, content, style in [0,1]. Returns (content_loss, style_loss) scalars."""
        gen = self._norm(gen_01); con = self._norm(content_01); sty = self._norm(style_01)
        x_g, x_c, x_s = gen, con, sty
        c_loss = 0.0; s_loss = 0.0
        for name, layer in self.vgg.named_children():
            x_g = layer(x_g); x_c = layer(x_c); x_s = layer(x_s)
            if name == self.CONTENT_LAYER:
                c_loss = c_loss + F.mse_loss(x_g, x_c)
            if name in self.STYLE_LAYERS:
                s_loss = s_loss + F.mse_loss(self._gram(x_g), self._gram(x_s))
        return c_loss, s_loss


class LPIPSMetric(nn.Module):
    """LPIPS. Default net='vgg' (research convention); --lpips_backbone alex for speed."""
    def __init__(self, device, net="vgg"):
        super().__init__()
        import lpips
        self.net = lpips.LPIPS(net=net).to(device).eval()
        self.backend_name = net
        for p in self.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, a_01, b_01):
        return self.net(a_01 * 2 - 1, b_01 * 2 - 1).view(-1).mean().item()


class SSIMMetric(nn.Module):
    """Pure-torch SSIM (no torchmetrics dependency). Matches train_grpo.SSIMReward."""
    def __init__(self, device, window_size: int = 11):
        super().__init__()
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
        w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0).expand(3, 1, window_size, window_size)
        self.register_buffer("window", w.contiguous())
        self.pad = window_size // 2
        self.to(device)

    @torch.no_grad()
    def forward(self, a_01, b_01):
        w, pad = self.window, self.pad
        mu_x = F.conv2d(a_01, w, padding=pad, groups=3)
        mu_y = F.conv2d(b_01, w, padding=pad, groups=3)
        mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
        sx  = F.conv2d(a_01.square(), w, padding=pad, groups=3) - mu_x2
        sy  = F.conv2d(b_01.square(), w, padding=pad, groups=3) - mu_y2
        sxy = F.conv2d(a_01 * b_01,   w, padding=pad, groups=3) - mu_xy
        num = (2 * mu_xy + self.C1) * (2 * sxy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sx + sy + self.C2)
        return (num / den).mean().item()


class DreamSimMetric(nn.Module):
    def __init__(self, device):
        super().__init__()
        from dreamsim import dreamsim
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
        self.eval()

    @torch.no_grad()
    def forward(self, a_01, b_01):
        if a_01.shape[-1] != 224:
            a_01 = F.interpolate(a_01, 224, mode="bilinear", align_corners=False)
        if b_01.shape[-1] != 224:
            b_01 = F.interpolate(b_01, 224, mode="bilinear", align_corners=False)
        return self.model(a_01, b_01).view(-1).mean().item()


class CLIPMetric(nn.Module):
    """CLIP-I cosine similarity (higher = more semantically similar)."""
    def __init__(self, device, local_dir: str = ""):
        super().__init__()
        self.backend = None
        if local_dir and os.path.isdir(local_dir):
            from transformers import CLIPModel
            self.model = CLIPModel.from_pretrained(local_dir).to(device).eval()
            self.backend = "local"
        else:
            from transformers import CLIPModel
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32").to(device).eval()
            self.backend = "hub"
        for p in self.parameters(): p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.to(device)

    @torch.no_grad()
    def _encode(self, x):
        x = F.interpolate(x, 224, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        vision_out = self.model.vision_model(pixel_values=x)
        pooled = vision_out.pooler_output if hasattr(vision_out, "pooler_output") else vision_out[1]
        return F.normalize(self.model.visual_projection(pooled).float(), dim=-1)

    @torch.no_grad()
    def forward(self, a_01, b_01):
        return (self._encode(a_01) * self._encode(b_01)).sum(dim=-1).mean().item()


# =========================== Datasets ======================================
PAIR_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # -> [-1, 1]
])


class PairedDataset(Dataset):
    """OmniStyle / ImagePulse style: target/<content&&style> triplet layout."""
    def __init__(self, root: str, sample_files: List[str], transform=PAIR_TRANSFORM):
        self.root = root
        self.transform = transform
        self.files = sample_files

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        content_name, style_raw = f.split("&&")
        style_name = style_raw[:-4] if style_raw.endswith(".png") and \
            os.path.exists(os.path.join(self.root, "style", style_raw[:-4])) else style_raw
        c = Image.open(os.path.join(self.root, "content", content_name)).convert("RGB")
        s = Image.open(os.path.join(self.root, "style",   style_name)).convert("RGB")
        t = Image.open(os.path.join(self.root, "target",  f)).convert("RGB")
        return self.transform(c), self.transform(s), self.transform(t), f


class ImagePulseDataset(Dataset):
    """ImagePulse: each triplet is a directory with content.png / style.png / target.png."""
    def __init__(self, root: str, dirs: List[str], transform=PAIR_TRANSFORM):
        self.root = root; self.transform = transform; self.dirs = dirs

    def __len__(self): return len(self.dirs)

    def __getitem__(self, idx):
        d = self.dirs[idx]
        c = Image.open(os.path.join(self.root, d, "content.png")).convert("RGB")
        s = Image.open(os.path.join(self.root, d, "style.png")).convert("RGB")
        t = Image.open(os.path.join(self.root, d, "target.png")).convert("RGB")
        return self.transform(c), self.transform(s), self.transform(t), d


class UnpairedDataset(Dataset):
    """COCO + WikiArt: random (content, style) pair, NO target. Returns None for target."""
    def __init__(self, content_paths: List[str], style_paths: List[str],
                 n_pairs: int, transform=PAIR_TRANSFORM, seed: int = 42):
        self.content_paths = content_paths
        self.style_paths   = style_paths
        self.transform = transform
        rng = random.Random(seed)
        self.pairs = [(rng.choice(content_paths), rng.choice(style_paths))
                      for _ in range(n_pairs)]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        c_path, s_path = self.pairs[idx]
        c = Image.open(c_path).convert("RGB")
        s = Image.open(s_path).convert("RGB")
        tag = os.path.basename(c_path).split(".")[0] + "__" + os.path.basename(s_path).split(".")[0]
        return self.transform(c), self.transform(s), tag


def _collect_paired_omnistyle(root: str, n: int, seed: int):
    target_dir = os.path.join(root, "target")
    # Handle symlink / variant casings similar to build_concat_dataset
    if not os.path.isdir(target_dir):
        for cand in ("OmniStyle-150K", "OmniStyle-150k"):
            real = os.path.join(root, cand)
            if os.path.isdir(real):
                os.symlink(os.path.abspath(real), target_dir); break
    all_files = [f for f in os.listdir(target_dir) if "&&" in f]
    rng = random.Random(seed); rng.shuffle(all_files)
    return PairedDataset(root, all_files[:n])


def _collect_paired_imagepulse(root: str, n: int, seed: int):
    dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    dirs = [d for d in dirs if
            os.path.isfile(os.path.join(root, d, "content.png")) and
            os.path.isfile(os.path.join(root, d, "style.png")) and
            os.path.isfile(os.path.join(root, d, "target.png"))]
    rng = random.Random(seed); rng.shuffle(dirs)
    return ImagePulseDataset(root, dirs[:n])


def _collect_unpaired_cocowiki(content_root: str, style_root: str, n: int, seed: int):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG")
    cs, ss = [], []
    for e in exts:
        cs.extend(glob.glob(os.path.join(content_root, "**", e), recursive=True))
        ss.extend(glob.glob(os.path.join(style_root,   "**", e), recursive=True))
    cs.sort(); ss.sort()
    assert len(cs) > 0 and len(ss) > 0, f"empty dir: content={len(cs)} style={len(ss)}"
    return UnpairedDataset(cs, ss, n_pairs=n, seed=seed)


# =========================== Eval loop =====================================
@torch.no_grad()
def evaluate(generator_fn, dataset: Dataset, metrics: dict, device,
             is_paired: bool, seed: int, top_k=900, top_p=0.96,
             save_imgs_dir: str = None, dataset_name: str = "",
             model_name: str = "model"):
    """generator_fn(c_pm1, s_pm1, i) -> gen_01 in [0,1]. Abstract over StyleVAR / AdaIN / etc."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    agg = {k: [] for k in ("style_loss", "content_loss",
                           "lpips_ref", "ssim_ref", "dreamsim_ref", "clip_ref",
                           "infer_time_sec")}
    agg["ref_target"] = "GT" if is_paired else "content"

    if save_imgs_dir:
        os.makedirs(save_imgs_dir, exist_ok=True)

    t_start_all = time.time()
    for i, batch in enumerate(loader):
        if is_paired:
            c_pm1, s_pm1, t_pm1, tag = batch
            t_pm1 = t_pm1.to(device); t_01 = (t_pm1 + 1) * 0.5
        else:
            c_pm1, s_pm1, tag = batch
            t_01 = None
        c_pm1 = c_pm1.to(device); s_pm1 = s_pm1.to(device)
        c_01 = (c_pm1 + 1) * 0.5; s_01 = (s_pm1 + 1) * 0.5

        # Inference (delegated)
        torch.cuda.synchronize(); t0 = time.time()
        gen_01 = generator_fn(c_pm1, s_pm1, c_01, s_01, i).clamp(0, 1)
        torch.cuda.synchronize(); agg["infer_time_sec"].append(time.time() - t0)

        # Metrics
        c_loss, s_loss = metrics["vgg"](gen_01, c_01, s_01)
        agg["content_loss"].append(c_loss.item())
        agg["style_loss"].append(s_loss.item())

        ref = t_01 if is_paired else c_01
        agg["lpips_ref"   ].append(metrics["lpips"   ](gen_01, ref))
        agg["ssim_ref"    ].append(metrics["ssim"    ](gen_01, ref))
        agg["dreamsim_ref"].append(metrics["dreamsim"](gen_01, ref))
        agg["clip_ref"    ].append(metrics["clip"    ](gen_01, ref))

        # Optionally save comparison image
        if save_imgs_dir and i < 8:  # save first 8 of each dataset
            _save_quad(save_imgs_dir, f"{i:03d}_{model_name}_{str(tag[0])[:40]}.png",
                       c_01[0], s_01[0], t_01[0] if is_paired else None, gen_01[0])

        if (i + 1) % 10 == 0:
            print(f"  [{model_name}/{dataset_name}] {i+1}/{len(dataset)} done", flush=True)

    # Summarize
    import statistics as st
    summary = {}
    for k, vals in agg.items():
        if k == "ref_target": continue
        if len(vals) == 0: continue
        summary[f"{k}_mean"]   = sum(vals) / len(vals)
        summary[f"{k}_median"] = st.median(vals)
        summary[f"{k}_std"]    = st.pstdev(vals) if len(vals) > 1 else 0.0
    summary["ref_target"] = agg["ref_target"]
    summary["n_samples"]  = len(agg["content_loss"])
    summary["dataset_name"] = dataset_name
    summary["model_name"] = model_name
    summary["total_time_sec"] = time.time() - t_start_all
    return summary, agg


def _save_quad(out_dir, fname, c_01, s_01, t_01, g_01):
    """Save a 1xN strip: content | style | (target?) | generated."""
    import torchvision.utils as vutils
    imgs = [c_01.cpu(), s_01.cpu()]
    if t_01 is not None: imgs.append(t_01.cpu())
    imgs.append(g_01.cpu())
    grid = vutils.make_grid(torch.stack(imgs, dim=0), nrow=len(imgs), padding=2)
    vutils.save_image(grid, os.path.join(out_dir, fname))


# =========================== CLI + orchestration ===========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     type=str, default="",
                   help="Path to the StyleVAR checkpoint to evaluate. "
                        "Accepts plain SFT / LoRA-wrapped full_state / merged GRPO ckpts / "
                        "LoRA-only ckpts (requires --base_ckpt). "
                        "Can be omitted if --skip_stylevar is set (AdaIN-only run).")
    p.add_argument("--base_ckpt", type=str, default="",
                   help="Required if --ckpt is a LoRA-only checkpoint (contains only "
                        "lora_A/lora_B keys). The base onto which the LoRA delta is applied. "
                        "Typically ckpt/sft-best.pth or ckpt/grpo-best.pth.")
    p.add_argument("--vae_ckpt", type=str, default="ckpt/vae_ch160v4096z32.pth")
    p.add_argument("--clip_local_dir", type=str, default="ckpt/clip-vit-base-patch32",
                   help="Local HF CLIP snapshot (leave empty to use HF hub)")

    # Dataset paths
    p.add_argument("--omnistyle_root",  type=str, default="data/OmniStyle-150k")
    p.add_argument("--imagepulse_root", type=str, default="data/ImagePulse")
    p.add_argument("--coco_root",       type=str, default="data/coco2017/images/train2017")
    p.add_argument("--wikiart_root",    type=str, default="data/wikiart")

    # Per-dataset sample counts (0 = skip that dataset)
    p.add_argument("--omnistyle_n",  type=int, default=50)
    p.add_argument("--imagepulse_n", type=int, default=50)
    p.add_argument("--cocowiki_n",   type=int, default=50)

    # Sampling / inference
    p.add_argument("--top_k",  type=int,   default=900)
    p.add_argument("--top_p",  type=float, default=0.96)
    p.add_argument("--seed",   type=int,   default=42)

    # Metrics config
    p.add_argument("--lpips_backbone", type=str, default="vgg", choices=["vgg", "alex", "squeeze"],
                   help="LPIPS backbone. vgg = research default; alex = training-time default (faster)")

    # Baselines
    p.add_argument("--also_adain", action="store_true",
                   help="Also evaluate AdaIN as a baseline on the same samples (for comparison)")
    p.add_argument("--adain_decoder", type=str, default="ckpt/adain_decoder.pth",
                   help="Path to the AdaIN decoder weights (download via download_adain.sh)")
    p.add_argument("--skip_stylevar", action="store_true",
                   help="Skip StyleVAR evaluation (e.g. to eval AdaIN alone)")

    # Output
    p.add_argument("--out_dir", type=str, default="eval_out")
    p.add_argument("--save_samples", action="store_true", default=True,
                   help="Save per-sample comparison grids (first 8 of each dataset)")

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); random.seed(args.seed)
    print(f"[eval] device={device}  seed={args.seed}")

    # ---- Load StyleVAR (unless skipped) ----
    model = None
    if not args.skip_stylevar:
        if not args.ckpt:
            raise RuntimeError("--ckpt is required unless --skip_stylevar is set.")
        model, vae = load_model(args.ckpt, args.vae_ckpt, device, base_ckpt_path=args.base_ckpt)

    # ---- Optionally load AdaIN baseline ----
    adain = None
    if args.also_adain:
        if not os.path.isfile(args.adain_decoder):
            print(f"[eval] WARNING: --also_adain set but decoder not found at {args.adain_decoder}")
            print(f"[eval]          Run `bash download_adain.sh` first, or pass --adain_decoder PATH")
            print(f"[eval]          Skipping AdaIN.")
        else:
            print(f"[eval] loading AdaIN decoder: {args.adain_decoder}")
            adain = AdaINBaseline(device, args.adain_decoder)

    # ---- Build metrics once ----
    print("[eval] building metrics...")
    metrics = {
        "vgg":      VGGStyleContentMetric(device),
        "lpips":    LPIPSMetric(device, net=args.lpips_backbone),
        "ssim":     SSIMMetric(device),
        "dreamsim": DreamSimMetric(device),
        "clip":     CLIPMetric(device, local_dir=args.clip_local_dir),
    }
    print(f"[eval] LPIPS backbone = {metrics['lpips'].backend_name}")

    # ---- Run per-dataset evaluation ----
    all_summaries = []

    dataset_specs = [
        ("OmniStyle",   args.omnistyle_n,   True,
         lambda: _collect_paired_omnistyle(args.omnistyle_root, args.omnistyle_n, args.seed)),
        ("ImagePulse",  args.imagepulse_n,  True,
         lambda: _collect_paired_imagepulse(args.imagepulse_root, args.imagepulse_n, args.seed)),
        ("COCO+WikiArt", args.cocowiki_n,   False,
         lambda: _collect_unpaired_cocowiki(args.coco_root, args.wikiart_root, args.cocowiki_n, args.seed)),
    ]

    # Generator closures
    def stylevar_gen(c_pm1, s_pm1, c_01, s_01, i):
        return model.autoregressive_infer(
            B=1, style_img=s_pm1, content_img=c_pm1,
            top_k=args.top_k, top_p=args.top_p, g_seed=args.seed + i,
        )

    def adain_gen(c_pm1, s_pm1, c_01, s_01, i):
        return adain.infer(c_01, s_01, alpha=1.0)

    models_to_eval = []
    if model is not None: models_to_eval.append(("StyleVAR", stylevar_gen))
    if adain is not None: models_to_eval.append(("AdaIN",    adain_gen))
    if not models_to_eval:
        raise RuntimeError("Nothing to evaluate: --skip_stylevar is set and --also_adain "
                            "either unset or decoder missing.")

    for name, n, is_paired, build_fn in dataset_specs:
        if n <= 0:
            print(f"\n[eval] SKIP {name} (n={n})"); continue
        print(f"\n[eval] === {name} ({'in-dist paired' if is_paired else 'OOD unpaired'}, n={n}) ===")
        try:
            dset = build_fn()
        except Exception as e:
            print(f"[eval] FAILED to build {name}: {e}")
            continue
        save_dir = os.path.join(args.out_dir, "samples", name.replace("+", "_")) \
                   if args.save_samples else None

        for mname, gen_fn in models_to_eval:
            print(f"\n  -- Model: {mname} --")
            summary, _raw = evaluate(gen_fn, dset, metrics, device,
                                      is_paired=is_paired, seed=args.seed,
                                      top_k=args.top_k, top_p=args.top_p,
                                      save_imgs_dir=save_dir,
                                      dataset_name=name, model_name=mname)
            all_summaries.append(summary)
            _print_summary(summary)

    # ---- Save combined report ----
    if all_summaries:
        csv_path = os.path.join(args.out_dir, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            keys = sorted({k for s in all_summaries for k in s.keys()})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in all_summaries: w.writerow(s)
        json_path = os.path.join(args.out_dir, "summary.json")
        with open(json_path, "w") as f:
            json.dump({"ckpt": args.ckpt, "summaries": all_summaries,
                       "lpips_backbone": args.lpips_backbone}, f, indent=2)
        print(f"\n[eval] saved {csv_path} and {json_path}")
        _print_comparative_table(all_summaries)


def _print_summary(s):
    print(f"  Samples: {s['n_samples']}  ref={s['ref_target']}  total_time={s['total_time_sec']:.1f}s")
    print(f"  infer_time/sample = {s['infer_time_sec_mean']:.3f}s")
    rows = [
        ("Style Loss ↓ (vs style)",   "style_loss"),
        ("Content Loss ↓ (vs content)","content_loss"),
        (f"LPIPS ↓ (vs {s['ref_target']})",       "lpips_ref"),
        (f"SSIM ↑ (vs {s['ref_target']})",        "ssim_ref"),
        (f"DreamSim ↓ (vs {s['ref_target']})",    "dreamsim_ref"),
        (f"CLIP sim ↑ (vs {s['ref_target']})",    "clip_ref"),
    ]
    for label, key in rows:
        m = s.get(f"{key}_mean")
        sd = s.get(f"{key}_std", 0.0)
        if m is not None:
            print(f"    {label:<40s} {m:.4f}  (±{sd:.4f})")


def _print_comparative_table(summaries):
    # Columns: (model, dataset). Group by this.
    cols = [(s["model_name"], s["dataset_name"]) for s in summaries]
    rows = [
        ("Style Loss ↓",       "style_loss_mean"),
        ("Content Loss ↓",     "content_loss_mean"),
        ("LPIPS ↓",            "lpips_ref_mean"),
        ("SSIM ↑",             "ssim_ref_mean"),
        ("DreamSim ↓",         "dreamsim_ref_mean"),
        ("CLIP sim ↑",         "clip_ref_mean"),
        ("infer s/sample",     "infer_time_sec_mean"),
    ]
    col_labels = [f"{m}/{d[:12]}" for (m, d) in cols]
    col_width = max(18, max(len(c) + 2 for c in col_labels))
    print("\n" + "="*(28 + col_width * len(cols)))
    print("  Cross-model × cross-dataset comparison")
    print("="*(28 + col_width * len(cols)))
    header = f"  {'Metric':<24s}" + "".join(f"{c:>{col_width}s}" for c in col_labels)
    print(header); print("-"*len(header))
    for label, key in rows:
        line = f"  {label:<24s}"
        for s in summaries:
            v = s.get(key)
            line += f"{'-':>{col_width}s}" if v is None else f"{v:>{col_width}.4f}"
        print(line)
    print("="*len(header))


if __name__ == "__main__":
    main()
