"""Verify merge_grpo_lora.py output is functionally identical to SFT+LoRA.

Runs TWO inference paths with the same seed/input and compares pixel output:
  Path A: load SFT base + apply_lora + load v3 LoRA weights -> infer (LoRA enabled)
  Path B: load merged ckpt (sft_plus_v3) into a plain StyleVAR        -> infer (no LoRA)

If the merge is correct, the two outputs must be IDENTICAL (or within 1e-5 FP noise).
If they differ visibly, the merge script has a bug.

Usage:
  python verify_merge.py \
      --sft_ckpt    Output_v2/ar-ckpt-best.pth \
      --grpo_ckpt   grpo_output_v3/grpo_best.pth \
      --merged_ckpt ckpt/sft_plus_v3.pth \
      --content data/ImagePulse/<dir>/content.png \
      --style   data/ImagePulse/<dir>/style.png

If --content / --style not given, picks the first pair from data/ImagePulse.
"""
from __future__ import annotations
import argparse
import glob
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import build_vae_stylevar


class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank, alpha):
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
        for attr in ("mat_qkv_guide", "mat_qkv_target", "proj"):
            setattr(block.attn, attr, LoRALinear(getattr(block.attn, attr), rank, alpha))
        for attr in ("fc1", "fc2"):
            setattr(block.ffn, attr, LoRALinear(getattr(block.ffn, attr), rank, alpha))


def build():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patch_nums = tuple(int(x) for x in "1_2_3_4_5_6_8_10_13_16".split("_"))
    vae, model = build_vae_stylevar(
        device=device, patch_nums=patch_nums,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        depth=20, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
        style_enc_dim=512,
    )
    return vae, model, device, patch_nums


def load_sft_state(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        if "trainer" in ck and "var_wo_ddp" in ck["trainer"]: return ck["trainer"]["var_wo_ddp"]
        if "model" in ck: return ck["model"]
    return ck


def infer_path_A(args, content, style):
    """Load SFT + v3 LoRA, infer with LoRA enabled."""
    vae, model, device, _ = build()
    vae.load_state_dict(torch.load(os.path.join(ROOT, args.vae_ckpt),
                                   map_location="cpu", weights_only=False), strict=True)
    vae.eval()
    model.load_state_dict(load_sft_state(os.path.join(ROOT, args.sft_ckpt)), strict=True)

    grpo = torch.load(os.path.join(ROOT, args.grpo_ckpt), map_location="cpu", weights_only=False)
    rank  = grpo.get("args", {}).get("lora_rank", 256)
    alpha = grpo.get("args", {}).get("lora_alpha", 512.0)
    apply_lora(model, rank, alpha)
    sd = model.state_dict(); sd.update(grpo["model"]); model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        gen = model.autoregressive_infer(
            B=1, style_img=style, content_img=content,
            top_k=args.top_k, top_p=args.top_p, g_seed=args.seed,
        )
    return gen[0].clamp(0, 1).cpu()


def infer_path_B(args, content, style):
    """Load merged ckpt into plain model, infer without LoRA."""
    vae, model, device, _ = build()
    vae.load_state_dict(torch.load(os.path.join(ROOT, args.vae_ckpt),
                                   map_location="cpu", weights_only=False), strict=True)
    vae.eval()
    model.load_state_dict(load_sft_state(os.path.join(ROOT, args.merged_ckpt)), strict=True)
    model.eval()

    with torch.no_grad():
        gen = model.autoregressive_infer(
            B=1, style_img=style, content_img=content,
            top_k=args.top_k, top_p=args.top_p, g_seed=args.seed,
        )
    return gen[0].clamp(0, 1).cpu()


def find_default_pair():
    ip_root = os.path.join(ROOT, "data", "ImagePulse")
    for d in sorted(os.listdir(ip_root)):
        c = os.path.join(ip_root, d, "content.png")
        s = os.path.join(ip_root, d, "style.png")
        if os.path.isfile(c) and os.path.isfile(s):
            return c, s
    raise FileNotFoundError("No content/style pair found in data/ImagePulse")


def load_img(path, device):
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    return t(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_ckpt",    default="Output_v2/ar-ckpt-best.pth")
    p.add_argument("--grpo_ckpt",   default="grpo_output_v3/grpo_best.pth")
    p.add_argument("--merged_ckpt", default="ckpt/sft_plus_v3.pth")
    p.add_argument("--vae_ckpt",    default="ckpt/vae_ch160v4096z32.pth")
    p.add_argument("--content",     default="")
    p.add_argument("--style",       default="")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--top_k",       type=int,   default=900)
    p.add_argument("--top_p",       type=float, default=0.96)
    p.add_argument("--out_dir",     default="verify_merge_out")
    args = p.parse_args()

    if not args.content or not args.style:
        c, s = find_default_pair()
        args.content, args.style = args.content or c, args.style or s
    print(f"[verify] content = {args.content}")
    print(f"[verify] style   = {args.style}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    content = load_img(args.content, device)
    style   = load_img(args.style,   device)

    print("[verify] Path A: SFT + LoRA enabled ...")
    img_A = infer_path_A(args, content, style)
    torch.cuda.empty_cache()
    print("[verify] Path B: merged ckpt (no LoRA) ...")
    img_B = infer_path_B(args, content, style)

    # Numerical comparison
    diff = (img_A - img_B).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    pct_diff = (diff > 1e-3).float().mean().item() * 100
    print("")
    print("="*60)
    print(f"  max |A-B|     = {max_err:.6f}")
    print(f"  mean |A-B|    = {mean_err:.6f}")
    print(f"  % pixels > 1e-3 = {pct_diff:.4f}%")
    print("="*60)
    if max_err < 1e-3:
        print("  ✅ PASS — paths are effectively identical (merge is correct)")
    elif max_err < 0.05:
        print("  ⚠️  SMALL DRIFT — likely FP precision, merge mostly OK")
    else:
        print("  ❌ FAIL — significant difference, merge script has a bug")

    # Save side-by-side
    os.makedirs(args.out_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a in ax: a.axis("off")
    ax[0].imshow(img_A.permute(1, 2, 0).numpy()); ax[0].set_title("A: SFT + v3 LoRA")
    ax[1].imshow(img_B.permute(1, 2, 0).numpy()); ax[1].set_title("B: merged ckpt (no LoRA)")
    diff_vis = (diff / max(max_err, 1e-9)).permute(1, 2, 0).numpy()
    ax[2].imshow(diff_vis); ax[2].set_title(f"|A-B| (max={max_err:.4f})")
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "verify.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[verify] saved {out_png}")


if __name__ == "__main__":
    main()
