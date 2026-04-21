# StyleVAR GRPO — Full Reference

A complete reference for the GRPO (Group Relative Policy Optimization) setup for StyleVAR: algorithm, techniques, pilot studies, bug history, and reproduction.

---

## 1. Overview

**Goal**: Fine-tune a supervised-trained StyleVAR (visual autoregressive style-transfer model) with RL, using DreamSim as reward against paired `(content, style, target)` triplets from OmniStyle + ImagePulse.

**Stack**:
- **Model**: StyleVAR ~721M params, 10 scales (1×1 → 16×16, L=680 tokens), 20 transformer blocks × (AdaLNCrossAttn + FFN), ResNet-18 style/content encoders, VQ-VAE (vocab=4096, Cvae=32).
- **Adaptation**: LoRA (rank=256, alpha=512, scaling=2) on `mat_qkv_guide / mat_qkv_target / proj / fc1 / fc2` — 131M trainable params (18.2%).
- **Reference policy**: LoRA-disabled forward (toggle, zero extra memory).
- **Reward**: `DreamSim(gen, target) × 5.0` (primary). Legacy multi-reward (LPIPS + VGG-Gram + SSIM + TV) available but discouraged.
- **Hardware**: single GPU, fp32 (no AMP — AMP caused old_lp≠new_lp instability in earlier runs).

---

## 2. GRPO Algorithm — VAR-specific formulation

Per training step:

```
1. Rollout: sample G trajectories (tokens across all 10 scales) from π_θ(a|s)
   with top_k=900, top_p=0.96.
2. Compute rewards R_i for each trajectory via DreamSim(gen_i, target).
3. Compute group-relative advantages:
     A_i = (R_i - mean_G(R)) / std_G(R)
4. Compute log-probs under 3 policies (teacher-forcing forward over the sampled tokens):
     old_lp   = log π_θ_old(a) — LoRA enabled, no_grad  (same θ as rollout)
     new_lp   = log π_θ(a)     — LoRA enabled, grad     (θ being updated)
     ref_lp   = log π_θ_ref(a) — LoRA DISABLED, no_grad (reference = base weights)
5. Per-token clipped surrogate:
     ratio = exp(new_lp - old_lp)
     L_policy = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
6. Per-token Schulman k3 KL (non-negative):
     L_kl = kl_coef * (exp(log_ratio_kl) - 1 - log_ratio_kl)
7. Weighted sum over tokens (PANW), mean over batch:
     loss = (L_policy · w_PANW).sum() + (L_kl · w_PANW).sum()
8. Backprop, clip grad, AdamW step.
```

**Single on-policy update per rollout** → ratio ≈ 1 at step 0 (sanity check: `old_lp == new_lp` exactly at step 0).

---

## 3. Key Techniques

### 3.1 PANW — Per-Action Normalization Weighting
From *"VAR RL Done Right"* (arXiv:2601.02256). VAR's 10 scales have **256× token imbalance** (1 token at 1×1 vs. 256 tokens at 16×16). Default GRPO mean makes fine scales dominate gradient, but coarse scales matter more for semantic/layout.

**Per-token weight**: `w_t = 1 / (h_t · w_t)^α / Z` where α=0.7 (paper recommends 0.6-0.8).

| Scale | Tokens | w_t (normalized) |
|-------|--------|------------------|
| 1×1 | 1 | 0.0337 |
| 2×2 | 4 | 0.0128 |
| 3×3 | 9 | 0.0072 |
| 4×4 | 16 | 0.0048 |
| 5×5 | 25 | 0.0035 |
| 6×6 | 36 | 0.0027 |
| 8×8 | 64 | 0.0018 |
| 10×10 | 100 | 0.0013 |
| 13×13 | 169 | 0.0009 |
| 16×16 | 256 | 0.0007 |

Coarse scales get ~60× more per-token weight than fine. Stabilized training and fixed the "reward improved but no visual difference" pathology.

### 3.2 LoRA Toggle Reference (no extra memory)
Reference policy = same model with `set_lora_enabled(False)`. When LoRA disabled, forward uses only `base_weight`; when enabled, forward uses `base_weight + (lora_B @ lora_A) · scaling`. Zero memory overhead vs. separate `ref_model` copy.

### 3.3 Iterative Merge (peak-triggered + KL-emergency)
After a confirmed improvement, merge LoRA delta into `base_weight` and reinit LoRA to zero. This resets KL to 0, allowing the policy to escape the current reference's pull.

**Safety rails** (all must pass for normal merge):
- `merge_cooldown=300` — min steps since last merge
- `merge_min_gain=0.05` — reward_ema improvement over last-merge baseline
- `merge_patience=50` — must hold new peak for this many steps
- `save_peak_lora=True` — merge uses PEAK snapshot, not current possibly-degraded weights

**KL emergency** (bypasses `merge_cooldown`):
- Triggered when `raw_kl = kl_loss / kl_coef > merge_kl_threshold=2.0`
- Uses separate short cooldown: `merge_emergency_cooldown=50`
- Restores to peak snapshot before baking

### 3.4 DreamSim reward (vs multi-metric)
Single DreamSim distance against paired target. Replaced earlier LPIPS+Gram+SSIM+TV composite because:
- Multi-metric weights are hard to balance (Runs 12-17 showed reward improved but images identical).
- Multi-metric is prone to reward hacking.
- DreamSim gives one holistic perceptual signal correlated with human judgment.

**Pilot 2 validated**: DreamSim(out, content) has ρ=0.64 correlation with DreamSim(out, GT); CLIP is better (ρ=0.79), but CLIP fails as a TRAINING reward — see §4.2.

### 3.5 Robust checkpoint loading
`_load_var_state_strict` auto-detects and handles 3 cases:
1. Plain SFT state_dict → strict load
2. LoRA-wrapped full_state → bake LoRA delta, then strict load
3. LoRA-only → **raise RuntimeError** (prevents the v3 disaster of training on random init)

Strict mode forces errors on any missing/unexpected key.

### 3.6 Reproducible RNG
`--seed 42` seeds torch/numpy/random globally. Rollout per-step seed:
```python
rollout_base_seed = args.seed * 1_000_003 + global_step * args.G
# each g_idx gets (base + g_idx), so within-group diversity preserved
```

---

## 4. Pilot Studies

### 4.1 Pilot 1 — Scale × Reward Gradient Sensitivity
**Script**: `pilot_scale_reward.py`
**Question**: For each reward (LPIPS / CLIP / DreamSim / StyleLoss / SSIM), which VAR scales receive strongest gradient?
**Method**: Differentiable "soft rollout" (softmax-weighted codebook instead of hard sampling) → backward the reward → measure `||∇_logits||` per scale.

**Status**: Partially working. Soft rollout has in-place-op issues in VAE decoder path (even with Phi monkey-patch). Used `torch.autograd.set_detect_anomaly(True)` for debugging. Can be revisited if needed for paper ablation; not blocking.

### 4.2 Pilot 2 — DreamSim content-vs-GT Correlation
**Script**: `pilot_dreamsim_proxy.py`
**Question**: Can `DreamSim(out, content)` substitute `DreamSim(out, GT)` when GT is unavailable (COCO+WikiArt)?
**Method**: N=200 paired triplets. Use other triplets' GTs as candidate outputs. Measure per-row Spearman correlation between content-based and GT-based distance rankings.

**Results**:
| Metric | mean ρ | median | %>0.5 | %>0.7 |
|--------|--------|--------|-------|-------|
| **CLIP** | **0.794** | 0.818 | **96%** | **80%** |
| LPIPS | 0.712 | 0.772 | 84% | 58% |
| DreamSim | 0.640 | 0.691 | 70% | 48% |

**Composite (α·content_dreamsim + (1-α)·style_gram) vs GT-DreamSim**:
- α=1.0 (pure content DreamSim): ρ=0.640
- α=0.5: ρ=0.605
- α=0.0 (pure style Gram): ρ=0.229

Adding style term DOESN'T help; VGG-Gram is weak as GT-proxy.

**Key finding**: CLIP has highest *ranking* consistency. Counter-intuitive because CLIP is "semantic, lossy" — but in style transfer, semantics are preserved while pixels change, so CLIP is stable across stylization variance.

**Caveat for v4 (see §5)**: Pilot 2 measures ranking consistency, NOT gradient quality. CLIP as GRPO reward FAILED despite high Pilot 2 score.

---

## 5. Training Run History

| Run | Base | Reward | Key Feature | Outcome |
|-----|------|--------|-------------|---------|
| 1-11 | SFT | LPIPS+Gram+SSIM | Early iterations | Code bugs |
| 12-14 | SFT | LPIPS+Gram+SSIM (1:1:1) | lr=5e-5 | KL blew up ~80 steps |
| 15-17 | SFT | LPIPS+Gram+SSIM (1:1:1) | lr=1e-5 | ~400 step window, KL drift; "reward up but images identical" |
| 18 | SFT | LPIPS+Gram+SSIM (5:0.3:2) | content-heavy + iter merge | Marginal improvement |
| **v2** | SFT | DreamSim | iterative merge + full_state saves | Real policy, saved as `grpo-best.pth` component |
| **v3** | v2 | DreamSim + PANW | `G=16`, `kl_coef=0.1`, `panw_alpha=0.7` | **Accidentally trained on random-init transformer** (strict=False bug) — still climbed -3.6 → -3.35 |
| **v4** | v2+v3 merged | **CLIP** | Iterative merge enabled | **Failed** — CLIP reward oscillated 0.80-0.88, R_ema regressed after step 55 |
| **v5** | SFT (current) | DreamSim + PANW + iterative | All bugs fixed | Fresh restart from proper SFT base |

### 5.1 v3's "Amazing" Curve (historical artifact)
Due to `strict=False` in var_ckpt load, v3 silently dropped v2's LoRA-wrapped transformer keys. Transformer ran at `init_weights` random init; only ResNet encoders + VAE loaded correctly. Yet GRPO+LoRA still climbed reward 0.25 in 800 steps. **Unintentional ablation** showing:
1. Algorithm robust to poor prior
2. Pretrained feature extractors (VAE + encoders) carry significant structural prior
3. LoRA (131M params) can learn substantial new behavior from scratch

The -3.6 → -3.35 gap is the *floor* of the algorithm; v5 starts at -1.25 (real SFT baseline) — that 2.35 delta quantifies what SFT actually provides.

### 5.2 v4's CLIP Reward Failure
Pilot 2 said CLIP has best ranking consistency (ρ=0.79). Tried CLIP(gen, target) as GRPO reward. Within 135 steps:
- CLIP cosine stuck in 0.80-0.88 (narrow range → small group variance → noisy advantages)
- R_ema peaked at step 55, then regressed
- Training effectively failed

**Lesson**: ranking consistency ≠ gradient signal quality. CLIP's narrow active range makes it a poor RL reward even though it's a great evaluation metric.

---

## 6. Bug History & Fixes

### 6.1 `strict=False` silent key drop (CRITICAL — fixed)
**Symptom**: v3 trained on random-init transformer blocks.
**Cause**: `model.load_state_dict(st, strict=False)` silently dropped LoRA-wrapped keys (`.base_weight`, `.lora_A`, `.lora_B`) when loading into a plain StyleVAR.
**Fix**: `_load_var_state_strict` in `train_grpo.py` — tries `strict=True` after auto-baking LoRA wrappers, refuses LoRA-only ckpts with loud error.
**Test**: `test_load_var_state_strict.py` — 6 cases all pass.

### 6.2 KL Emergency Blocked by Regular Cooldown (fixed)
**Symptom**: v5 run-1 showed KL exploding at step 580-600 but no emergency merge triggered; OOM at step 620.
**Cause**: emergency merge required the same `merge_cooldown=300` as normal merges, so between step 314 (first merge) and 614 no merge was possible regardless of KL level.
**Fix**: separate `--merge_emergency_cooldown=50` bypasses the regular cooldown. Emergency is now actually emergent.

### 6.3 Adaptive KL (all variants rejected — kept as historical lesson)
Three variants tried:
- Bidirectional: collapsed kl_coef to 0.0009 → KL divergence unbounded
- Upward-only: ratcheted kl_coef to 0.22 → killed all learning
- Target-based: oscillated

**Current**: fixed `kl_coef=0.1` + KL spike skip (`raw_kl_per_G > 5.0`) + emergency merge.

### 6.4 BatchNorm in Training Mode (fixed long ago)
**Symptom**: `old_lp != new_lp` at step 0, ratio ≠ 1.
**Cause**: `model.train()` used batch statistics for BN in ResNet encoders, causing stochastic outputs.
**Fix**: `model.eval()` throughout training (loss gradients still flow; only BN/Dropout are frozen).

### 6.5 VAE in-place `transpose_()` + autograd (fixed)
`quant.py` uses `.transpose_()` in places that crash backward if embedding weight has `requires_grad=True`. Must freeze VAE **completely** via `p.requires_grad_(False)`.

### 6.6 OOM from cache fragmentation (mitigated)
After many `torch.cuda.empty_cache()` calls (e.g. during KL spike skips), 24GB / 48GB fills with fragmented segments. Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in run script.

### 6.7 Reward hacking with multi-metric (avoided)
Runs 12-17 with `(lam_content, lam_style, lam_ssim)` weights tuned by hand: reward improved but images unchanged. Switched to DreamSim single reward.

### 6.8 Hyperparameter Sensitivities (documented)
| Param | Safe range | Comment |
|-------|-----------|---------|
| `lr` | 1e-5 (stable), 3e-5 (aggressive) | 5e-5 killed run at step ~80 |
| `kl_coef` | 0.05-0.1 fixed | Don't adapt |
| `grad_clip` | 1.0 | Works, 0.5 more conservative |
| `G` | 8-16 | Serial, no extra memory |
| `batch_size` | 12-16 | fp32 + bs=16 ≈ 45GB on 48GB GPU |
| `panw_alpha` | 0.6-0.8 | Paper default 0.7 |
| `top_k / top_p` | 900 / 0.96 | Kept at SFT inference defaults |
| `clip_eps` | 0.2 | Standard PPO |
| `merge_cooldown` | 300 | Prevent merge thrashing |
| `merge_min_gain` | 0.05 | Don't merge marginal noise |
| `merge_patience` | 50 | Confirm peak isn't noise |
| `merge_kl_threshold` | 2.0 | Emergency only |
| `merge_emergency_cooldown` | 50 | Short enough to rescue |

---

## 7. File Layout

### Training + inference
```
train_grpo.py             — main training loop; CLIPReward and DreamSimReward classes
                            _load_var_state_strict (safe loader)
                            rollout_generate, compute_logprobs_single
                            compute_group_advantages, merge_lora_and_reinit
                            snapshot_lora, restore_lora
infer_grpo.py             — load SFT + GRPO LoRA, generate SFT vs GRPO comparison
merge_grpo_lora.py        — standalone: bake v3-style LoRA into a v2-style base
                            (for external ckpt consolidation)
quick_infer.ipynb         — visual comparison notebook (reference flow)
```

### Run scripts
```
run_grpo.sh               — v3 config (DreamSim + PANW, no iterative merge)
run_grpo_v4.sh            — v4 config (CLIP reward + iterative merge) — DEPRECATED
run_grpo_v5.sh            — v5 config (DreamSim + PANW + iterative merge)  ← CURRENT
                            defaults: --var_ckpt ckpt/sft-best.pth, seed=42
```

### Pilots + verification
```
pilot_scale_reward.py     — Pilot 1: scale × reward gradient heatmap
pilot_dreamsim_proxy.py   — Pilot 2: DreamSim content-vs-GT ranking correlation
run_pilots.sh             — runs both pilots
test_load_var_state_strict.py — unit tests for the strict loader
```

### Reward/model assets
```
ckpt/sft-best.pth                 — SFT base  (canonical name; was Output_v2/ar-ckpt-best.pth)
ckpt/grpo-best.pth                — v2+v3 merged base (plain state_dict; was v2_plus_v3.pth)
ckpt/vae_ch160v4096z32.pth        — VAE (frozen)
ckpt/clip-vit-base-patch32/       — CLIP weights (HF snapshot)
ckpt/dreamsim/                    — DreamSim cache (DINO-ViT-B/16 + LoRA adapter)
```

### Data
```
data/OmniStyle-150k/              — paired triplets
data/ImagePulse/                  — paired triplets
data/coco2017/images/train2017/   — content images (unpaired setup, not used in v5)
data/wikiart/                     — style images (unpaired setup, not used in v5)
```

---

## 8. How to Reproduce

### 8.1 Fresh v5 run from SFT
```bash
pkill -f train_grpo 2>/dev/null
bash run_grpo_v5.sh
tail -f grpo_v5.log
```

Startup sanity (first few log lines should show):
```
[GRPO] RNG seeded with seed=42
SFT ckpt: .../ckpt/sft-best.pth
[GRPO] StyleVAR loaded from .../ckpt/sft-best.pth
[PANW] alpha=0.7, weights per scale: 1x1=0.0337, ...
```

Step 0 DEBUG output should show:
- `rewards: [16 diverse values]` — group has variance (exploration working)
- `ratio: mean=1.0000, std=0.0000` — old_lp exactly equals new_lp
- `log_ratio: abs_max=0.0` — no drift yet
- `advantage: std≈1.0` — group-normalized correctly

### 8.2 Continue from v2+v3 merged
```bash
bash run_grpo_v5.sh --var_ckpt ckpt/grpo-best.pth
```

### 8.3 Resume a v5 run
```bash
bash run_grpo_v5.sh --resume grpo_output_v5/grpo_latest.pth
```

### 8.4 Merge a new LoRA into the current best
After a successful v5 run:
```bash
python merge_grpo_lora.py \
    --sft_ckpt  ckpt/sft-best.pth \            # or ckpt/grpo-best.pth to stack
    --grpo_ckpt grpo_output_v5/grpo_best.pth \
    --out       ckpt/grpo-best.pth             # overwrite rolling pointer
```

### 8.5 Inference comparison
```bash
python infer_grpo.py \
    --sft_ckpt  Output_v2/ar-ckpt-best.pth \
    --grpo_ckpt grpo_output_v5/grpo_best.pth \
    --num 4 --out comparison.png
```

### 8.6 Run Pilots
```bash
bash run_pilots.sh          # both
bash run_pilots.sh p2       # just pilot 2 (faster, no gradient through rollout)
```

---

## 9. Current State (2026-04-19)

- **Active**: v5 fresh from SFT, `seed=42`, DreamSim reward, PANW α=0.7, iterative merge (normal cooldown=300, emergency cooldown=50).
- **Early trajectory** (step 1-90): R_ema flat at ~-1.26 (per-step R varies -1.0 to -1.5 from batch variance). EMA stability suggests proper on-policy update; slow movement is expected because SFT baseline is already strong.
- **Decision point**: let v5 run to step 300-400 and re-evaluate. If R_ema doesn't trend up, tune `lr` (1e-5 → 3e-5) and/or `kl_coef` (0.1 → 0.05).

---

## 10. Open Questions / Future Work

1. **Pilot 1 soft rollout**: fix remaining in-place op in VAE decoder to unblock the scale-reward heatmap. Needed for paper motivation figure.
2. **COCO+WikiArt unpaired**: switching to decoupled content/style data would eliminate the ImagePulse leakage (style image often shares semantics with content). Requires re-evaluating reward (no GT). CLIP(out, content) is a candidate — but see §5.2, CLIP didn't work as training reward in paired setting. Need further investigation.
3. **Iterative GRPO systematic sweep**: given all bugs fixed, re-examine whether merge helps on top of PANW. v5 will tell.
4. **Scale-aware multi-reward**: based on Pilot 1 outcome, route each reward to its "native" scales (content→coarse, style→fine). Aspirational.
5. **DreamSim scale saturation**: at R_ema≈-1.25 (DreamSim distance ≈ 0.25), we may be near the metric's discrimination floor. Consider ensembling DreamSim + another metric, or switching once saturated.

---

## 11. What This Project Does NOT Use (deliberate)

- ❌ **Mixed precision / AMP** — caused old_lp ≠ new_lp issues in earlier runs; fp32 throughout
- ❌ **Adaptive kl_coef** — all three variants failed
- ❌ **Multi-reward weighted sum** — reward hacking + hard-to-balance weights
- ❌ **Fixed interval iterative merge** — can ratchet reward down
- ❌ **CFG during rollout** — doubles batch, no clear RL gain
- ❌ **Model.train() mode** — BN corrupts sampling; model.eval() throughout
- ❌ **CLIP as GRPO reward** — pilot 2 said good, v4 empirically failed
