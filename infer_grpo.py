"""Quick inference with a GRPO-trained LoRA checkpoint.

Usage:
  python infer_grpo.py --grpo_ckpt grpo_output/grpo_best.pth --num 8
  python infer_grpo.py --grpo_ckpt grpo_output/grpo_best.pth --sft_only  # SFT baseline comparison
"""

import os, sys, glob, random, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import build_vae_stylevar


# ---- LoRA (copied from train_grpo.py, inference-only) ----
class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.base_weight = base_linear.weight
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grpo_ckpt", type=str, required=True, help="Path to GRPO checkpoint")
    p.add_argument("--sft_ckpt", type=str, default="Output/ar-ckpt-best.pth")
    p.add_argument("--vae_ckpt", type=str, default="ckpt/vae_ch160v4096z32.pth")
    p.add_argument("--sft_only", action="store_true", help="Generate with SFT model only (no LoRA)")
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=900)
    p.add_argument("--top_p", type=float, default=0.96)
    p.add_argument("--out", type=str, default="grpo_infer_results.png")
    p.add_argument("--data_dir", type=str, default="",
                   help="Data dir with content/style subdirs (default: auto-detect)")
    return p.parse_args()


def find_data(root):
    """Auto-detect data directory with content/style images."""
    candidates = [
        os.path.join(root, "data", "OmniStyle-150K"),
        os.path.join(root, "data", "ImagePulse"),
    ]
    for d in candidates:
        c = os.path.join(d, "content")
        s = os.path.join(d, "style")
        if os.path.isdir(c) and os.path.isdir(s):
            return c, s
    # Fallback: COCO + WikiArt
    c = os.path.join(root, "data", "coco2017", "images", "train2017")
    s = os.path.join(root, "data", "wikiart")
    if os.path.isdir(c) and os.path.isdir(s):
        return c, s
    raise FileNotFoundError("Cannot find content/style data directories")


def collect_images(directory, max_files=10000):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        if len(files) >= max_files:
            break
    return files


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Build model ----
    patch_nums = tuple(int(x) for x in "1_2_3_4_5_6_8_10_13_16".split("_"))
    vae, model = build_vae_stylevar(
        device=device, patch_nums=patch_nums,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        depth=20, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
        style_enc_dim=512,
    )

    # ---- Load VAE ----
    vae_path = os.path.join(ROOT, args.vae_ckpt)
    vae.load_state_dict(torch.load(vae_path, map_location="cpu"), strict=True)
    vae.eval()

    # ---- Load SFT base ----
    sft_path = os.path.join(ROOT, args.sft_ckpt)
    sft_ckpt = torch.load(sft_path, map_location="cpu")
    if "trainer" in sft_ckpt and "var_wo_ddp" in sft_ckpt["trainer"]:
        sft_state = sft_ckpt["trainer"]["var_wo_ddp"]
    elif "model" in sft_ckpt:
        sft_state = sft_ckpt["model"]
    else:
        sft_state = sft_ckpt
    model.load_state_dict(sft_state, strict=True)
    print(f"[SFT] Loaded from {sft_path}")

    if not args.sft_only:
        # ---- Load GRPO LoRA ----
        grpo_ckpt = torch.load(args.grpo_ckpt, map_location="cpu")
        grpo_args = grpo_ckpt.get("args", {})
        rank = grpo_args.get("lora_rank", 256)
        alpha = grpo_args.get("lora_alpha", 512.0)
        step = grpo_ckpt.get("step", "?")

        apply_lora(model, rank, alpha)
        model_sd = model.state_dict()
        model_sd.update(grpo_ckpt["model"])
        model.load_state_dict(model_sd)
        print(f"[GRPO] Loaded LoRA from {args.grpo_ckpt} (step={step}, rank={rank})")

    model.eval()

    # ---- Load data ----
    if args.data_dir:
        content_dir = os.path.join(args.data_dir, "content")
        style_dir = os.path.join(args.data_dir, "style")
    else:
        content_dir, style_dir = find_data(ROOT)

    content_files = collect_images(content_dir)
    style_files = collect_images(style_dir)
    print(f"[Data] {len(content_files)} content, {len(style_files)} style")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    random.seed(args.seed)
    N = min(args.num, len(content_files), len(style_files))
    c_samples = random.sample(content_files, N)
    s_samples = random.sample(style_files, N)

    # ---- Generate ----
    fig, axes = plt.subplots(N, 3, figsize=(12, 4 * N))
    if N == 1:
        axes = axes[None, :]

    mode_str = "SFT only" if args.sft_only else f"GRPO (step={step})"
    axes[0, 0].set_title("Content", fontsize=14)
    axes[0, 1].set_title("Style", fontsize=14)
    axes[0, 2].set_title(f"Generated ({mode_str})", fontsize=14)

    with torch.no_grad():
        for i in range(N):
            content = transform(Image.open(c_samples[i]).convert("RGB")).unsqueeze(0).to(device)
            style = transform(Image.open(s_samples[i]).convert("RGB")).unsqueeze(0).to(device)

            gen = model.autoregressive_infer(
                B=1, style_img=style, content_img=content,
                top_k=args.top_k, top_p=args.top_p, g_seed=args.seed + i,
            )

            c_show = content[0].mul(0.5).add(0.5).clamp(0, 1)
            s_show = style[0].mul(0.5).add(0.5).clamp(0, 1)
            g_show = gen[0].clamp(0, 1)

            for ax in axes[i]:
                ax.axis("off")
            axes[i, 0].imshow(c_show.permute(1, 2, 0).cpu().numpy())
            axes[i, 1].imshow(s_show.permute(1, 2, 0).cpu().numpy())
            axes[i, 2].imshow(g_show.permute(1, 2, 0).cpu().numpy())

    plt.tight_layout()
    out_path = os.path.join(ROOT, args.out)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()

    # ---- Also save SFT comparison if running GRPO ----
    if not args.sft_only:
        print("\nGenerating SFT baseline for comparison...")
        # Disable LoRA = SFT baseline
        from train_grpo import set_lora_enabled
        set_lora_enabled(model, False)

        fig2, axes2 = plt.subplots(N, 4, figsize=(16, 4 * N))
        if N == 1:
            axes2 = axes2[None, :]
        axes2[0, 0].set_title("Content", fontsize=14)
        axes2[0, 1].set_title("Style", fontsize=14)
        axes2[0, 2].set_title("SFT", fontsize=14)
        axes2[0, 3].set_title(f"GRPO (step={step})", fontsize=14)

        set_lora_enabled(model, False)
        sft_gens = []
        with torch.no_grad():
            for i in range(N):
                content = transform(Image.open(c_samples[i]).convert("RGB")).unsqueeze(0).to(device)
                style = transform(Image.open(s_samples[i]).convert("RGB")).unsqueeze(0).to(device)
                gen_sft = model.autoregressive_infer(
                    B=1, style_img=style, content_img=content,
                    top_k=args.top_k, top_p=args.top_p, g_seed=args.seed + i,
                )
                sft_gens.append(gen_sft)

        set_lora_enabled(model, True)
        with torch.no_grad():
            for i in range(N):
                content = transform(Image.open(c_samples[i]).convert("RGB")).unsqueeze(0).to(device)
                style = transform(Image.open(s_samples[i]).convert("RGB")).unsqueeze(0).to(device)
                gen_grpo = model.autoregressive_infer(
                    B=1, style_img=style, content_img=content,
                    top_k=args.top_k, top_p=args.top_p, g_seed=args.seed + i,
                )

                c_show = content[0].mul(0.5).add(0.5).clamp(0, 1)
                s_show = style[0].mul(0.5).add(0.5).clamp(0, 1)

                for ax in axes2[i]:
                    ax.axis("off")
                axes2[i, 0].imshow(c_show.permute(1, 2, 0).cpu().numpy())
                axes2[i, 1].imshow(s_show.permute(1, 2, 0).cpu().numpy())
                axes2[i, 2].imshow(sft_gens[i][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
                axes2[i, 3].imshow(gen_grpo[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy())

        plt.tight_layout()
        cmp_path = os.path.join(ROOT, args.out.replace(".png", "_comparison.png"))
        plt.savefig(cmp_path, dpi=150, bbox_inches="tight")
        print(f"Comparison saved to {cmp_path}")
        plt.close()


if __name__ == "__main__":
    main()
