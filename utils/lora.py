"""Shared LoRA + checkpoint-loading helpers for StyleVAR GRPO.

Single source of truth for:
  - LoRALinear class (drop-in wrap over a frozen nn.Linear)
  - apply_lora(model, rank, alpha)       — inject adapters
  - set_lora_enabled(model, enabled)     — toggle policy vs reference forward
  - snapshot_lora / restore_lora         — CPU snapshot for peak-restore merges
  - _bake_lora_into_plain                — fold LoRA delta into base_weight
  - _apply_lora_to_plain_state           — overlay LoRA-only ckpt onto plain base
  - _extract_state_dict                  — unwrap {"model": ...} / {"trainer": ...} etc.
  - load_var_state_strict                — safe loader for all 3 ckpt formats
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================== LoRALinear ====================================
class LoRALinear(nn.Module):
    """A frozen base_linear wrapped with a LoRA residual.

    effective_weight = base_weight + (lora_B @ lora_A) * scaling, with scaling = alpha / rank.
    When _enabled=False, the LoRA residual is skipped (useful for reference-policy forwards).
    """
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


# =========================== Injection / toggle ============================
LORA_ATTN_ATTRS = ("mat_qkv_guide", "mat_qkv_target", "proj")
LORA_FFN_ATTRS  = ("fc1", "fc2")


def apply_lora(model, rank: int, alpha: float) -> List[nn.Parameter]:
    """Replace every attention + FFN Linear in model.blocks with a LoRALinear.
    Returns the list of trainable LoRA parameters (lora_A, lora_B pairs)."""
    params: List[nn.Parameter] = []
    for block in model.blocks:
        for attr in LORA_ATTN_ATTRS:
            old = getattr(block.attn, attr)
            new = LoRALinear(old, rank, alpha)
            setattr(block.attn, attr, new)
            params.extend([new.lora_A, new.lora_B])
        for attr in LORA_FFN_ATTRS:
            old = getattr(block.ffn, attr)
            new = LoRALinear(old, rank, alpha)
            setattr(block.ffn, attr, new)
            params.extend([new.lora_A, new.lora_B])
    return params


def set_lora_enabled(model, enabled: bool):
    """Toggle LoRA path on/off. enabled=False → reference policy (base only)."""
    for block in model.blocks:
        for attr in LORA_ATTN_ATTRS:
            m = getattr(block.attn, attr)
            if isinstance(m, LoRALinear):
                m._enabled = enabled
        for attr in LORA_FFN_ATTRS:
            m = getattr(block.ffn, attr)
            if isinstance(m, LoRALinear):
                m._enabled = enabled


def merge_lora_and_reinit(model) -> List[nn.Parameter]:
    """Bake LoRA delta into base_weight, reset lora_A/B so KL restarts at 0.
    Returns the new list of trainable LoRA parameters for the optimizer."""
    params: List[nn.Parameter] = []
    for block in model.blocks:
        for attr in LORA_ATTN_ATTRS + LORA_FFN_ATTRS:
            owner = block.attn if attr in LORA_ATTN_ATTRS else block.ffn
            m = getattr(owner, attr)
            if isinstance(m, LoRALinear):
                with torch.no_grad():
                    m.base_weight.add_((m.lora_B @ m.lora_A) * m.scaling)
                    nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
                    m.lora_B.zero_()
                params.extend([m.lora_A, m.lora_B])
    return params


def snapshot_lora(model) -> Dict[int, tuple]:
    """CPU snapshot of all LoRA tensors, keyed by module id."""
    snap = {}
    for block in model.blocks:
        for attr in LORA_ATTN_ATTRS:
            m = getattr(block.attn, attr)
            if isinstance(m, LoRALinear):
                snap[id(m)] = (m.lora_A.detach().clone().cpu(),
                                m.lora_B.detach().clone().cpu())
        for attr in LORA_FFN_ATTRS:
            m = getattr(block.ffn, attr)
            if isinstance(m, LoRALinear):
                snap[id(m)] = (m.lora_A.detach().clone().cpu(),
                                m.lora_B.detach().clone().cpu())
    return snap


def restore_lora(model, snap: Dict[int, tuple]):
    """Write snapshotted LoRA tensors back into the model (in place)."""
    for block in model.blocks:
        for attr in LORA_ATTN_ATTRS:
            m = getattr(block.attn, attr)
            if isinstance(m, LoRALinear) and id(m) in snap:
                A, B = snap[id(m)]
                with torch.no_grad():
                    m.lora_A.copy_(A.to(m.lora_A.device))
                    m.lora_B.copy_(B.to(m.lora_B.device))
        for attr in LORA_FFN_ATTRS:
            m = getattr(block.ffn, attr)
            if isinstance(m, LoRALinear) and id(m) in snap:
                A, B = snap[id(m)]
                with torch.no_grad():
                    m.lora_A.copy_(A.to(m.lora_A.device))
                    m.lora_B.copy_(B.to(m.lora_B.device))


# =========================== Checkpoint helpers ============================
def _extract_state_dict(raw):
    """Unwrap common checkpoint wrappings to get the underlying state_dict."""
    if not isinstance(raw, dict):
        return raw
    if "trainer" in raw and isinstance(raw["trainer"], dict) and "var_wo_ddp" in raw["trainer"]:
        return raw["trainer"]["var_wo_ddp"]
    for key in ("model", "state_dict", "model_state_dict", "weights", "params"):
        v = raw.get(key)
        if isinstance(v, dict) and len(v) > 0 and isinstance(next(iter(v.values())), torch.Tensor):
            return v
    if len(raw) > 0 and isinstance(next(iter(raw.values())), torch.Tensor):
        return raw
    if len(raw) == 1:
        v = next(iter(raw.values()))
        if isinstance(v, dict) and len(v) > 0 and isinstance(next(iter(v.values())), torch.Tensor):
            return v
    raise RuntimeError(
        f"Could not find a state_dict in the ckpt. Top-level keys: {list(raw.keys())[:10]}."
    )


def bake_lora_into_plain(state: dict, rank: int, alpha: float) -> dict:
    """Convert a LoRA-wrapped state_dict (base_weight + lora_A + lora_B)
    to a plain one (weight) by folding (lora_B @ lora_A) * scaling into base_weight.
    No-op if no LoRA keys are present."""
    scaling = alpha / rank
    lora_prefixes = {k[:-len(".lora_A")] for k in state.keys() if k.endswith(".lora_A")}
    if not lora_prefixes:
        return state
    out = {}
    for k, v in state.items():
        matched = False
        for p in lora_prefixes:
            if k == p + ".base_weight":
                lA = state[p + ".lora_A"]; lB = state[p + ".lora_B"]
                out[p + ".weight"] = (v + (lB @ lA) * scaling).contiguous()
                matched = True; break
            if k == p + ".base_bias":
                out[p + ".bias"] = v; matched = True; break
            if k in (p + ".lora_A", p + ".lora_B"):
                matched = True; break
        if not matched:
            out[k] = v
    return out


def apply_lora_to_plain_state(base_state: dict, lora_state: dict,
                               rank: int, alpha: float):
    """Fold a LoRA-only state_dict onto a PLAIN base state_dict.
    Returns (new_plain_state_dict, n_modules_applied)."""
    scaling = alpha / rank
    out = dict(base_state)
    lora_prefixes = {k[:-len(".lora_A")] for k in lora_state.keys() if k.endswith(".lora_A")}
    n_applied = 0; missing = []
    for p in lora_prefixes:
        weight_key = p + ".weight"
        if weight_key not in out:
            missing.append(weight_key); continue
        lA = lora_state[p + ".lora_A"]; lB = lora_state[p + ".lora_B"]
        out[weight_key] = (out[weight_key] + (lB @ lA) * scaling).contiguous()
        n_applied += 1
    if missing:
        raise RuntimeError(f"LoRA prefixes have no matching plain weight in base: {missing[:3]}...")
    return out, n_applied


def load_var_state_strict(model, st: dict, full_ckpt: dict = None, base_state: dict = None):
    """Load a StyleVAR state_dict with strict=True. Handles 3 cases:

      1. Plain (`*.weight`)               → strict load
      2. LoRA-wrapped full_state (`*.base_weight`/`*.lora_A/B`)
         → bake LoRA into base, then strict load
      3. LoRA-only (only `*.lora_A/B`)    → REQUIRES base_state; fold onto base, load

    Raises RuntimeError on any missing/unexpected key, preventing silent key drop
    (the v3 disaster where transformer trained on random init for 800 steps).
    """
    full_ckpt = full_ckpt or {}
    has_base_weight = any(k.endswith(".base_weight") for k in st.keys())
    has_lora_A      = any(k.endswith(".lora_A")      for k in st.keys())

    if has_lora_A and not has_base_weight:
        # Case 3: LoRA-only
        if base_state is None:
            raise RuntimeError(
                "[load_var] ckpt is LoRA-only (no base_weight / weight keys). "
                "Supply a base_state (e.g. SFT ckpt) to stack the LoRA onto."
            )
        args_in = full_ckpt.get("args", {}) if isinstance(full_ckpt, dict) else {}
        rank  = args_in.get("lora_rank", 256)
        alpha = args_in.get("lora_alpha", 512.0)
        st, n = apply_lora_to_plain_state(base_state, st, rank, alpha)
        print(f"[load_var] applied {n} LoRA modules onto supplied base state")

    elif has_base_weight:
        # Case 2: LoRA-wrapped full_state — bake LoRA into plain weights
        args_in = full_ckpt.get("args", {}) if isinstance(full_ckpt, dict) else {}
        rank  = args_in.get("lora_rank", 256)
        alpha = args_in.get("lora_alpha", 512.0)
        print(f"[load_var] detected LoRA-wrapped ckpt; baking "
              f"(rank={rank}, alpha={alpha}) into plain weights")
        st = bake_lora_into_plain(st, rank, alpha)

    # Case 1: plain — fall through to strict load
    model.load_state_dict(st, strict=True)
