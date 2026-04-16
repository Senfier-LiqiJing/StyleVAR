"""Pilot Study 2: Can DreamSim(out, content) substitute DreamSim(out, GT)?

Goal: validate whether a content-only DreamSim reward gives the same *ranking*
signal as a GT-based DreamSim reward. If yes, we can drop GT and move to
unpaired data (COCO + WikiArt) without losing the supervision quality.

Method (NO training, NO inference — pure metric analysis on paired dataset):
  1. Sample N triplets (content_i, style_i, gt_i) from OmniStyle + ImagePulse.
  2. Use other triplets' gt_j as "candidate outputs" for triplet i.
     This simulates "what if the model produced gt_j for content_i?".
  3. For each triplet i and candidate j (j != i), compute:
       d_gt     = DreamSim(gt_j, gt_i)        — true quality (ideal reward signal)
       d_content= DreamSim(gt_j, content_i)   — proposed content-only reward
  4. Per-row Spearman correlation between d_gt and d_content (across j).
     High correlation (> 0.7) = content-only DreamSim is a viable proxy.
  5. Also evaluate CLIP-I and LPIPS as alternative content-only rewards.
  6. Save: histogram of Spearman rho, scatter plots, summary CSV.

Usage (on the GPU machine with paired data):
  python pilot_dreamsim_proxy.py \
      --old_data_path data/OmniStyle-150k \
      --new_data_path data/ImagePulse \
      --n_triplets 200 \
      --out_dir pilot_results

Output:
  pilot_results/dreamsim_proxy_spearman.png    — histogram
  pilot_results/dreamsim_proxy_scatter.png     — scatter (single triplet example)
  pilot_results/dreamsim_proxy_summary.csv     — per-metric mean/median/quartile stats
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models as tv_models

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# =========================== Metrics =======================================
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
        """a, b in [0,1], (B, 3, H, W). Returns (B,) distances (higher = less similar)."""
        if a_01.shape[-1] != 224:
            a_01 = F.interpolate(a_01, 224, mode='bilinear', align_corners=False)
        if b_01.shape[-1] != 224:
            b_01 = F.interpolate(b_01, 224, mode='bilinear', align_corners=False)
        return self.model(a_01, b_01).view(-1)


class CLIPMetric(nn.Module):
    """1 - cosine similarity (distance), higher = less similar."""
    def __init__(self, device):
        super().__init__()
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            self.model = model.to(device).eval()
            self.backend = "open_clip"
        except Exception:
            from transformers import CLIPModel
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
            self.backend = "hf"
        for p in self.parameters(): p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    @torch.no_grad()
    def _encode(self, img_01):
        x = F.interpolate(img_01, 224, mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        if self.backend == "open_clip":
            f = self.model.encode_image(x)
        else:
            f = self.model.get_image_features(pixel_values=x)
        return F.normalize(f.float(), dim=-1)

    @torch.no_grad()
    def forward(self, a_01, b_01):
        return 1.0 - (self._encode(a_01) * self._encode(b_01)).sum(dim=-1)


class LPIPSMetric(nn.Module):
    def __init__(self, device):
        super().__init__()
        import lpips
        self.net = lpips.LPIPS(net="alex").to(device).eval()
        for p in self.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, a_01, b_01):
        a_pm1 = a_01 * 2 - 1; b_pm1 = b_01 * 2 - 1
        return self.net(a_pm1, b_pm1).view(-1)


class VGGGramMetric(nn.Module):
    """Style distance via Gram matrices (higher = less similar style)."""
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
        self.to(device).eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _gram(x):
        B, C, H, W = x.shape
        f = x.view(B, C, H * W)
        return torch.bmm(f, f.transpose(1, 2)) / (C * H * W)

    @torch.no_grad()
    def forward(self, a_01, b_01):
        a = (a_01 - self.mean) / self.std
        b = (b_01 - self.mean) / self.std
        d = a.new_zeros(a.shape[0])
        xa, xb = a, b
        for sl in self.slices:
            xa = sl(xa); xb = sl(xb)
            d = d + (self._gram(xa) - self._gram(xb)).square().mean(dim=(1, 2))
        return d


# =========================== Core ==========================================
def load_triplets(args, device):
    """Load N paired triplets and stack tensors on device.
    Returns content (N,3,H,W), style (N,3,H,W), gt (N,3,H,W), all in [0,1]."""
    from utils.data import build_concat_dataset
    _, dataset, _ = build_concat_dataset(
        old_data_path=args.old_data_path,
        new_data_path=args.new_data_path,
        new_data_tar_dir="",
        final_reso=args.image_size,
    )
    print(f"[Data] paired dataset size={len(dataset)}")

    rng = np.random.RandomState(args.seed)
    all_idx = rng.permutation(len(dataset))[:args.n_triplets]

    contents, styles, gts = [], [], []
    for k, idx in enumerate(all_idx):
        # StyleTransferDataset returns (target, style, content) in [-1, 1]
        target_pm1, style_pm1, content_pm1 = dataset[int(idx)]
        contents.append(content_pm1); styles.append(style_pm1); gts.append(target_pm1)
        if (k + 1) % 50 == 0: print(f"  loaded {k+1}/{len(all_idx)}")
    content_pm1 = torch.stack(contents).to(device)
    style_pm1   = torch.stack(styles).to(device)
    gt_pm1      = torch.stack(gts).to(device)
    return (content_pm1 + 1) * 0.5, (style_pm1 + 1) * 0.5, (gt_pm1 + 1) * 0.5


def pairwise_distance(metric: nn.Module, A: torch.Tensor, B: torch.Tensor,
                      bs: int = 32) -> np.ndarray:
    """Compute metric(A[i], B[j]) for all i,j. Returns (N_A, N_B).
    Uses .expand (no copy) for the repeated image to keep memory flat."""
    NA, NB = A.shape[0], B.shape[0]
    out = np.zeros((NA, NB), dtype=np.float32)
    for i in range(NA):
        for s in range(0, NB, bs):
            e = min(s + bs, NB)
            a_rep = A[i:i+1].expand(e - s, -1, -1, -1)
            out[i, s:e] = metric(a_rep, B[s:e]).detach().cpu().numpy()
    return out


def per_row_spearman(d_gt: np.ndarray, d_content: np.ndarray) -> np.ndarray:
    """For each row i, Spearman correlation between d_gt[i, j] and d_content[i, j]
    across j != i. Returns (N,) correlations."""
    from scipy.stats import spearmanr
    N = d_gt.shape[0]
    rhos = np.zeros(N)
    mask_tmpl = np.ones(N, dtype=bool)
    for i in range(N):
        m = mask_tmpl.copy(); m[i] = False
        rho, _ = spearmanr(d_gt[i, m], d_content[i, m])
        rhos[i] = rho if not np.isnan(rho) else 0.0
    return rhos


def run_pilot(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Data ----
    content_01, style_01, gt_01 = load_triplets(args, device)
    N = content_01.shape[0]
    print(f"[Data] N={N} triplets, image_size={args.image_size}")

    # ---- Metrics ----
    print("[Metrics] building DreamSim / CLIP / LPIPS / VGG-Gram ...")
    metrics = {
        "DreamSim": DreamSimMetric(device),
        "CLIP":     CLIPMetric(device),
        "LPIPS":    LPIPSMetric(device),
    }

    # ---- Compute pairwise distance matrices ----
    # For each metric M and each triplet i, we need:
    #   d_gt[i, j]      = M(gt_j, gt_i)
    #   d_content[i, j] = M(gt_j, content_i)
    results = {}
    for mname, metric in metrics.items():
        print(f"[{mname}] computing NxN matrices (N={N})...")
        t0 = time.time()
        d_gt      = pairwise_distance(metric, gt_01,      gt_01, bs=args.bs)
        d_content = pairwise_distance(metric, content_01, gt_01, bs=args.bs)
        # Interpretation: row i = "candidate outputs gt_j are compared against
        # triplet i's reference (gt_i or content_i)".
        # So d_gt[i, j] = metric(reference_gt_i, candidate_gt_j) via metric(A,B)
        # where A = gt[i] repeated, B = gt[j]. With our code: pairwise_distance(m, gt, gt)
        # returns out[i, j] = m(gt[i], gt[j]). Correct by symmetry for all metrics used.
        print(f"  done in {time.time()-t0:.1f}s")

        rhos = per_row_spearman(d_gt, d_content)
        results[mname] = {"d_gt": d_gt, "d_content": d_content, "rho": rhos}
        print(f"  Spearman: mean={rhos.mean():.3f}  median={np.median(rhos):.3f}  "
              f"%>0.5={100*(rhos>0.5).mean():.0f}%  %>0.7={100*(rhos>0.7).mean():.0f}%")

    # ---- Save summary ----
    _save_summary_csv(os.path.join(args.out_dir, "dreamsim_proxy_summary.csv"), results)
    _plot_spearman_hist(os.path.join(args.out_dir, "dreamsim_proxy_spearman.png"), results)
    _plot_scatter_example(os.path.join(args.out_dir, "dreamsim_proxy_scatter.png"), results, example_row=0)

    # ---- Bonus: test simple composite (content + style) for DreamSim case ----
    if args.also_composite:
        print("[Composite] testing alpha * d_content + (1-alpha) * d_style ...")
        vgg = VGGGramMetric(device)
        d_style_to_gt = pairwise_distance(vgg, style_01, gt_01, bs=args.bs)
        # d_style_to_gt[i, j] = style_dist(style_i, gt_j)
        # For each alpha, compute per-row Spearman vs d_gt (using DreamSim as anchor)
        d_gt_dream  = results["DreamSim"]["d_gt"]
        d_cont_dream= results["DreamSim"]["d_content"]
        def _norm(x): return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        composite = {}
        for a in alphas:
            proxy = a * _norm(d_cont_dream) + (1 - a) * _norm(d_style_to_gt)
            rhos_c = per_row_spearman(d_gt_dream, proxy)
            composite[a] = rhos_c
            print(f"  alpha={a}: mean_rho={rhos_c.mean():.3f}")
        _save_composite_csv(os.path.join(args.out_dir, "dreamsim_proxy_composite.csv"), composite)

    print(f"[Done] Results saved to {args.out_dir}")


def _save_summary_csv(path, results):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean_rho", "median_rho", "q25", "q75", "pct_gt_0.5", "pct_gt_0.7"])
        for mname, r in results.items():
            rho = r["rho"]
            w.writerow([mname,
                        f"{rho.mean():.4f}", f"{np.median(rho):.4f}",
                        f"{np.quantile(rho, 0.25):.4f}", f"{np.quantile(rho, 0.75):.4f}",
                        f"{100*(rho>0.5).mean():.2f}", f"{100*(rho>0.7).mean():.2f}"])


def _save_composite_csv(path, composite):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha_content", "mean_rho_vs_d_gt_dreamsim", "median_rho"])
        for a, rhos in composite.items():
            w.writerow([a, f"{rhos.mean():.4f}", f"{np.median(rhos):.4f}"])


def _plot_spearman_hist(path, results):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for mname, r in results.items():
        ax.hist(r["rho"], bins=30, alpha=0.5, label=f"{mname} (mean={r['rho'].mean():.2f})", density=True)
    ax.axvline(0.5, color='k', linestyle='--', alpha=0.5, label='rho=0.5')
    ax.axvline(0.7, color='r', linestyle='--', alpha=0.5, label='rho=0.7')
    ax.set_xlabel("Per-triplet Spearman(d_metric(cand, content), d_metric(cand, gt))")
    ax.set_ylabel("Density")
    ax.set_title("Content-only ranking vs GT-based ranking — per-triplet Spearman")
    ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _plot_scatter_example(path, results, example_row=0):
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for k, (mname, r) in enumerate(results.items()):
        i = example_row
        N = r["d_gt"].shape[0]
        mask = np.arange(N) != i
        ax = axes[0, k]
        ax.scatter(r["d_gt"][i, mask], r["d_content"][i, mask], s=10, alpha=0.5)
        ax.set_xlabel(f"{mname}(cand, gt)")
        ax.set_ylabel(f"{mname}(cand, content)")
        ax.set_title(f"{mname}  (triplet idx={i}, rho={r['rho'][i]:.2f})")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# =========================== CLI ===========================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--old_data_path", type=str, default="data/OmniStyle-150k")
    p.add_argument("--new_data_path", type=str, default="data/ImagePulse")
    p.add_argument("--n_triplets", type=int, default=200)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--bs",         type=int, default=32, help="batch size for metric calls")
    p.add_argument("--out_dir",    type=str, default="pilot_results")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--also_composite", action="store_true",
                   help="Also test alpha * DreamSim(cand,content) + (1-alpha) * VGGGram(cand, style) proxy")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_pilot(args)


if __name__ == "__main__":
    main()
