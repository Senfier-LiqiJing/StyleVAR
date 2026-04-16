"""Merge a GRPO LoRA checkpoint into the SFT base and save a single consolidated
checkpoint that can be used as `--sft_ckpt` / `--var_ckpt` for a fresh GRPO run
(e.g. v4 will init a new zero LoRA on top of this merged policy).

Usage:
  python merge_grpo_lora.py \
      --sft_ckpt  Output_v2/ar-ckpt-best.pth \
      --grpo_ckpt grpo_output_v3/grpo_best.pth \
      --out       ckpt/sft_plus_v3.pth
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import build_vae_stylevar


# ---- LoRA (duplicated from train_grpo.py to keep this script standalone) ----
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


# ---- Core merge logic ----
def extract_sft_state(sft_ckpt):
    """SFT ckpts may be raw state_dicts or wrapped in {trainer:{...}} / {model:...}."""
    if isinstance(sft_ckpt, dict):
        if "trainer" in sft_ckpt and "var_wo_ddp" in sft_ckpt["trainer"]:
            return sft_ckpt["trainer"]["var_wo_ddp"]
        if "model" in sft_ckpt:
            return sft_ckpt["model"]
    return sft_ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_ckpt",  required=True,
                        help="SFT base checkpoint (e.g. Output_v2/ar-ckpt-best.pth)")
    parser.add_argument("--grpo_ckpt", required=True,
                        help="GRPO LoRA checkpoint to merge in (e.g. grpo_output_v3/grpo_best.pth)")
    parser.add_argument("--vae_ckpt",  default="ckpt/vae_ch160v4096z32.pth")
    parser.add_argument("--out",       required=True,
                        help="Output path for the merged SFT+LoRA checkpoint")
    parser.add_argument("--device",    default="cpu",
                        help="Device for merge (cpu is safe; cuda:0 is faster)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[merge] device={device}")

    # ---- Build model (empty) ----
    patch_nums = tuple(int(x) for x in "1_2_3_4_5_6_8_10_13_16".split("_"))
    vae, model = build_vae_stylevar(
        device=device, patch_nums=patch_nums,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        depth=20, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
        style_enc_dim=512,
    )
    vae.load_state_dict(torch.load(os.path.join(ROOT, args.vae_ckpt), map_location="cpu"),
                        strict=True)

    # ---- Load SFT base ----
    sft_raw = torch.load(os.path.join(ROOT, args.sft_ckpt) if not os.path.isabs(args.sft_ckpt)
                         else args.sft_ckpt, map_location="cpu")
    sft_state = extract_sft_state(sft_raw)
    model.load_state_dict(sft_state, strict=True)
    print(f"[merge] SFT base loaded: {args.sft_ckpt}  ({len(sft_state)} tensors)")

    # ---- Apply LoRA using rank/alpha from the GRPO ckpt ----
    grpo = torch.load(os.path.join(ROOT, args.grpo_ckpt) if not os.path.isabs(args.grpo_ckpt)
                      else args.grpo_ckpt, map_location="cpu")
    gargs = grpo.get("args", {}) if isinstance(grpo, dict) else {}
    rank  = gargs.get("lora_rank", 256)
    alpha = gargs.get("lora_alpha", 512.0)
    step  = grpo.get("step", "?") if isinstance(grpo, dict) else "?"
    apply_lora(model, rank, alpha)
    print(f"[merge] LoRA applied (rank={rank}, alpha={alpha})")

    # ---- Load GRPO LoRA weights ----
    grpo_state = grpo.get("model", grpo) if isinstance(grpo, dict) else grpo
    model_sd = model.state_dict()
    # Only update keys that exist; warn on unexpected / missing LoRA keys
    common = {k: v for k, v in grpo_state.items() if k in model_sd}
    unexpected = [k for k in grpo_state.keys() if k not in model_sd]
    model_sd.update(common)
    model.load_state_dict(model_sd, strict=True)
    print(f"[merge] GRPO LoRA loaded: step={step}  "
          f"({len(common)} keys matched, {len(unexpected)} unexpected keys ignored)")
    if unexpected[:5]:
        print(f"[merge] unexpected sample: {unexpected[:5]}")

    # ---- Merge: bake (lora_B @ lora_A) * scaling into base_weight ----
    n_merged = 0
    with torch.no_grad():
        for block in model.blocks:
            for owner, attrs in [(block.attn, ("mat_qkv_guide", "mat_qkv_target", "proj")),
                                 (block.ffn, ("fc1", "fc2"))]:
                for name in attrs:
                    m = getattr(owner, name)
                    if isinstance(m, LoRALinear):
                        delta = (m.lora_B @ m.lora_A) * m.scaling
                        m.base_weight.add_(delta)
                        # Zero LoRA so subsequent save doesn't double-count
                        m.lora_A.zero_()
                        m.lora_B.zero_()
                        n_merged += 1
    print(f"[merge] merged {n_merged} LoRA modules into base weights")

    # ---- Extract plain state dict (rename & drop LoRA artifacts) ----
    full_sd = model.state_dict()
    merged_state = {}
    dropped = 0
    for k, v in full_sd.items():
        if k.endswith(".lora_A") or k.endswith(".lora_B"):
            dropped += 1
            continue
        if k.endswith(".base_weight"):
            new_k = k[:-len(".base_weight")] + ".weight"
        elif k.endswith(".base_bias"):
            new_k = k[:-len(".base_bias")] + ".bias"
        else:
            new_k = k
        merged_state[new_k] = v.detach().cpu().clone()
    print(f"[merge] plain state_dict built: {len(merged_state)} tensors "
          f"(dropped {dropped} LoRA keys)")

    # ---- Sanity: roundtrip load on a fresh plain model ----
    del model
    _, fresh = build_vae_stylevar(
        device=torch.device("cpu"), patch_nums=patch_nums,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        depth=20, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
        style_enc_dim=512,
    )
    fresh.load_state_dict(merged_state, strict=True)
    print(f"[merge] roundtrip sanity check: OK (loaded as plain StyleVAR)")

    # ---- Save ----
    out_path = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"model": merged_state,
                "note": f"merged {args.sft_ckpt} + LoRA from {args.grpo_ckpt} (step={step}, "
                        f"rank={rank}, alpha={alpha})"},
               out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[merge] saved -> {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
