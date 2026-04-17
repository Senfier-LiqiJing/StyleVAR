"""Verify _load_var_state_strict handles all 3 cases correctly.

Test cases:
  1. Plain SFT-style state_dict    → strict load succeeds
  2. LoRA-wrapped full_state dict  → bake LoRA, strict load succeeds, weights == base + delta
  3. LoRA-only state_dict          → RuntimeError (silent-failure prevention)
  4. Partial/broken state_dict     → RuntimeError (missing keys)
  5. Extra junk key                → RuntimeError (unexpected key)
"""
import math
import os
import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---- Minimal plain "block" that mimics StyleVAR's Linear layout ----
class ToyAttn(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.mat_qkv_guide  = nn.Linear(dim, dim, bias=False)
        self.mat_qkv_target = nn.Linear(dim, dim, bias=False)
        self.proj           = nn.Linear(dim, dim, bias=True)


class ToyFFN(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2, bias=True)
        self.fc2 = nn.Linear(dim * 2, dim, bias=True)


class ToyBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.attn = ToyAttn(dim)
        self.ffn  = ToyFFN(dim)


class ToyStyleVAR(nn.Module):
    """Stand-in for StyleVAR matching the Linear structure that LoRA wraps."""
    def __init__(self, n_blocks=3, dim=64):
        super().__init__()
        self.blocks = nn.ModuleList([ToyBlock(dim) for _ in range(n_blocks)])


# ---- LoRA (copied to not import from train_grpo, which would load heavy deps) ----
class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank, alpha):
        super().__init__()
        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.base_weight  = base_linear.weight
        self.has_bias = base_linear.bias is not None
        if self.has_bias:
            self.base_bias = base_linear.bias
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank


def wrap_with_lora(model, rank, alpha):
    for block in model.blocks:
        for a in ("mat_qkv_guide", "mat_qkv_target", "proj"):
            setattr(block.attn, a, LoRALinear(getattr(block.attn, a), rank, alpha))
        for a in ("fc1", "fc2"):
            setattr(block.ffn, a, LoRALinear(getattr(block.ffn, a), rank, alpha))


# ---- Inline copy of _load_var_state_strict so we don't need to import model code ----
def load_var_state_strict(model, st, full_ckpt=None):
    full_ckpt = full_ckpt or {}
    has_base_weight = any(k.endswith(".base_weight") for k in st.keys())
    has_lora_A      = any(k.endswith(".lora_A")      for k in st.keys())

    if has_lora_A and not has_base_weight:
        raise RuntimeError("[load_var] Refused: LoRA-only keys; would train on random init.")

    if has_base_weight:
        args_in_ckpt = full_ckpt.get("args", {}) if isinstance(full_ckpt, dict) else {}
        rank  = args_in_ckpt.get("lora_rank", 256)
        alpha = args_in_ckpt.get("lora_alpha", 512.0)
        scaling = alpha / rank
        prefixes = {k[:-len(".lora_A")] for k in st.keys() if k.endswith(".lora_A")}
        baked = {}
        for k, v in st.items():
            matched = False
            for p in prefixes:
                if k == p + ".base_weight":
                    lA = st[p + ".lora_A"]; lB = st[p + ".lora_B"]
                    baked[p + ".weight"] = (v + (lB @ lA) * scaling).contiguous()
                    matched = True; break
                if k == p + ".base_bias":
                    baked[p + ".bias"] = v; matched = True; break
                if k == p + ".lora_A" or k == p + ".lora_B":
                    matched = True; break
            if not matched:
                baked[k] = v
        st = baked

    model.load_state_dict(st, strict=True)


# ============================== Tests ======================================

def ok(msg): print(f"  \033[92m✓\033[0m {msg}")
def fail(msg): print(f"  \033[91m✗\033[0m {msg}"); sys.exit(1)


def test_case_1_plain():
    """Plain SFT ckpt → strict load succeeds, weights match."""
    print("\n[Case 1] Plain SFT state_dict")
    gold = ToyStyleVAR()
    state = {k: v.clone() for k, v in gold.state_dict().items()}
    target = ToyStyleVAR()  # fresh init
    load_var_state_strict(target, state)
    for k in state:
        if not torch.equal(state[k], dict(target.state_dict())[k]):
            fail(f"mismatch on {k}")
    ok("plain ckpt loads strictly, weights identical")


def test_case_2_lora_wrapped():
    """LoRA-wrapped full_state → bake+load, effective weight == base + lora_delta."""
    print("\n[Case 2] LoRA-wrapped full_state (v2-style)")
    # Build a trained "v2" model: plain → wrap → set lora_B non-zero (simulate training)
    v2 = ToyStyleVAR()
    torch.manual_seed(123)
    for p in v2.parameters(): p.data = torch.randn_like(p) * 0.1
    wrap_with_lora(v2, rank=4, alpha=8.0)
    # simulate that post-merge training continued: set lora_B to non-zero
    for m in v2.modules():
        if isinstance(m, LoRALinear):
            m.lora_B.data = torch.randn_like(m.lora_B) * 0.01

    # Save v2 state (LoRA-wrapped keys)
    v2_state = {k: v.clone() for k, v in v2.state_dict().items()}
    v2_args  = {"args": {"lora_rank": 4, "lora_alpha": 8.0}}

    # Compute the expected effective weight per-block (index in key to avoid collisions)
    expected = {}
    scaling = 8.0 / 4
    for b_idx, block in enumerate(v2.blocks):
        for owner_name, owner, attrs in [("attn", block.attn, ("mat_qkv_guide", "mat_qkv_target", "proj")),
                                           ("ffn",  block.ffn,  ("fc1", "fc2"))]:
            for a in attrs:
                m = getattr(owner, a)
                if isinstance(m, LoRALinear):
                    eff = m.base_weight + (m.lora_B @ m.lora_A) * scaling
                    expected[(b_idx, owner_name, a, "weight")] = eff.detach().clone()
                    if m.has_bias:
                        expected[(b_idx, owner_name, a, "bias")] = m.base_bias.detach().clone()

    # Load into a fresh plain model
    target = ToyStyleVAR()
    load_var_state_strict(target, v2_state, v2_args)

    # Verify each Linear's weight matches the expected (base + lora_delta)
    checked = 0
    for i, block in enumerate(target.blocks):
        for owner_name, owner in [("attn", block.attn), ("ffn", block.ffn)]:
            for name, child in owner.named_children():
                if isinstance(child, nn.Linear):
                    key_w = (i, owner_name, name, "weight")
                    exp_w = expected.get(key_w)
                    if exp_w is None:
                        fail(f"expected weight missing for block{i}.{owner_name}.{name}")
                    if not torch.allclose(child.weight, exp_w, atol=1e-6):
                        max_err = (child.weight - exp_w).abs().max().item()
                        fail(f"weight mismatch at block{i}.{owner_name}.{name}: max_err={max_err}")
                    if child.bias is not None:
                        exp_b = expected.get((i, owner_name, name, "bias"))
                        if exp_b is None or not torch.allclose(child.bias, exp_b):
                            fail(f"bias mismatch at block{i}.{owner_name}.{name}")
                    checked += 1
    ok(f"LoRA-wrapped ckpt baked correctly; {checked} Linear weights match base+lora_delta (all blocks)")


def test_case_3_lora_only():
    """LoRA-only state_dict → should raise RuntimeError (the v3 bug guard)."""
    print("\n[Case 3] LoRA-only state_dict (the v3 bug case)")
    v2 = ToyStyleVAR()
    wrap_with_lora(v2, rank=4, alpha=8.0)
    full = v2.state_dict()
    lora_only = {k: v for k, v in full.items() if "lora_" in k}

    target = ToyStyleVAR()
    try:
        load_var_state_strict(target, lora_only)
        fail("expected RuntimeError for LoRA-only ckpt")
    except RuntimeError as e:
        if "LoRA-only" in str(e) or "Refused" in str(e):
            ok(f"correctly refused LoRA-only ckpt: {e!s}")
        else:
            fail(f"wrong error: {e!s}")


def test_case_4_missing_keys():
    """Incomplete plain ckpt → strict must raise."""
    print("\n[Case 4] Incomplete state_dict (missing some keys)")
    gold = ToyStyleVAR()
    state = {k: v.clone() for k, v in gold.state_dict().items()}
    # Remove one key
    dropped = list(state.keys())[5]
    del state[dropped]
    target = ToyStyleVAR()
    try:
        load_var_state_strict(target, state)
        fail("expected RuntimeError for missing keys")
    except RuntimeError as e:
        if "Missing key" in str(e):
            ok(f"strict raised on missing keys (removed: {dropped})")
        else:
            fail(f"wrong error: {e!s}")


def test_case_5_unexpected_key():
    """Plain ckpt + an extra junk key → strict must raise."""
    print("\n[Case 5] Unexpected key")
    gold = ToyStyleVAR()
    state = {k: v.clone() for k, v in gold.state_dict().items()}
    state["blocks.0.attn.JUNK_KEY"] = torch.randn(4)
    target = ToyStyleVAR()
    try:
        load_var_state_strict(target, state)
        fail("expected RuntimeError for unexpected key")
    except RuntimeError as e:
        if "Unexpected key" in str(e) or "unexpected" in str(e).lower():
            ok("strict raised on unexpected key")
        else:
            fail(f"wrong error: {e!s}")


def test_case_6_v3_scenario():
    """End-to-end: simulate the v3 bug — LoRA-wrapped v2 ckpt being loaded
    the OLD way (strict=False) silently drops keys; NEW way refuses or bakes."""
    print("\n[Case 6] v3 regression scenario (original strict=False bug)")
    # Build v2 (LoRA wrapped, non-zero LoRA delta simulating post-merge training)
    v2 = ToyStyleVAR()
    torch.manual_seed(0)
    for p in v2.parameters(): p.data = torch.randn_like(p) * 0.1
    wrap_with_lora(v2, rank=4, alpha=8.0)
    for m in v2.modules():
        if isinstance(m, LoRALinear):
            m.lora_B.data = torch.randn_like(m.lora_B) * 0.03
    v2_state = {k: v.clone() for k, v in v2.state_dict().items()}
    v2_full_ckpt = {"args": {"lora_rank": 4, "lora_alpha": 8.0}, "model": v2_state}

    # OLD behavior: strict=False would silently leave transformer blocks at fresh init
    old_target = ToyStyleVAR()
    old_init_weight = {k: v.clone() for k, v in old_target.state_dict().items()}
    old_target.load_state_dict(v2_state, strict=False)  # silent
    # Check: did weights actually load? (expected: NO, because keys are .base_weight)
    unchanged_count = sum(1 for k in old_init_weight
                          if torch.equal(old_init_weight[k], dict(old_target.state_dict())[k]))
    total = len(old_init_weight)
    print(f"    OLD strict=False: {unchanged_count}/{total} plain keys left at random init (this was the bug)")
    if unchanged_count != total:
        fail(f"expected all {total} plain keys unchanged, got {unchanged_count}")

    # NEW behavior: strict load with auto-bake
    new_target = ToyStyleVAR()
    load_var_state_strict(new_target, v2_state, v2_full_ckpt)
    # Check: weights now MATCH the effective v2 policy
    match_count = 0
    for block_i, (b_new, b_src) in enumerate(zip(new_target.blocks, v2.blocks)):
        for owner_name in ("attn", "ffn"):
            owner_new = getattr(b_new, owner_name)
            owner_src = getattr(b_src, owner_name)
            for name, child_new in owner_new.named_children():
                child_src = getattr(owner_src, name)
                if isinstance(child_new, nn.Linear) and isinstance(child_src, LoRALinear):
                    eff = child_src.base_weight + (child_src.lora_B @ child_src.lora_A) * child_src.scaling
                    if torch.allclose(child_new.weight, eff, atol=1e-6):
                        match_count += 1
                    else:
                        max_err = (child_new.weight - eff).abs().max().item()
                        fail(f"block{block_i}.{owner_name}.{name}: max_err={max_err}")
    ok(f"NEW strict+bake: {match_count} Linear weights match v2's effective policy (vs OLD which left them at random init)")


if __name__ == "__main__":
    print("="*60)
    print("Verifying _load_var_state_strict")
    print("="*60)
    test_case_1_plain()
    test_case_2_lora_wrapped()
    test_case_3_lora_only()
    test_case_4_missing_keys()
    test_case_5_unexpected_key()
    test_case_6_v3_scenario()
    print("\n" + "="*60)
    print("\033[92mAll 6 cases passed.\033[0m")
    print("="*60)
