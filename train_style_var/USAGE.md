# StyleVAR Fine-Tuning Usage

`train_style_var/fine_tune.py` reuses the original VAR argument parser (`utils.arg_util.Args`), so all switches from the base project work here as well. This document explains how to launch the script, how to disable flash-attention/xFormers, and what each argument controls.

## Launch Examples

### Single GPU / Debug
```bash
python train_style_var/fine_tune.py \
  --data-path ./dataset \
  --origin-ckpt-dir ./origin_checkpoints \
  --local-out-dir-path ./local_output/stylevar_ft \
  --tb-log-dir-path ./local_output/stylevar_ft/tb \
  --ep 5 --bs 16 --workers 4 --fuse 0
```

### Multi-GPU (8 GPUs via torchrun)
```bash
torchrun --nproc_per_node=8 train_style_var/fine_tune.py \
  --data-path /path/to/dataset \
  --origin-ckpt-dir /path/to/origin_checkpoints \
  --vanilla-ckpt-path /path/to/var_d20.pth \
  --vae-ckpt-path /path/to/vae_ch160v4096z32.pth \
  --exp-name style_var_ft \
  --bs 1024 --workers 12 --ac 128 \
  --depth 20 --fp16 1 --fuse 0 \
  --pn 1_2_3_4_5_6_8_10_13_16 --patch-size 16 \
  --mid-reso 1.125 --hflip 0 \
  --sche lin1 --tblr 5e-4 --tclip 2.0
```

> **Flash-Attention / xFormers:** These are governed by `--fuse`. Set `--fuse 0` (and `--afuse 0`) when those libraries are unavailable. Set `--fuse 1` only if flash-attn/xFormers/fused ops are installed.

## Argument Reference

All of the following flags can be passed to `train_style_var/fine_tune.py` (defaults come from `utils/arg_util.Args`).

### Paths
- `--data-path`: Dataset root containing `content/style/target` (default `/home/OmniStyle-150K`).
- `--origin-ckpt-dir`: Directory holding fallback checkpoints (`origin_checkpoints/` by default).
- `--clean-ckpt-path`: Optional fine-tuned StyleVAR checkpoint for initialization.
- `--vanilla-ckpt-path`: Vanilla VAR checkpoint (default `/home/PML-Project/checkpoints/var_d20.pth`).
- `--vae-ckpt-path`: Pretrained tokenizer checkpoint (default `/home/PML-Project/checkpoints/vae_ch160v4096z32.pth`).
- `--local-out-dir-path`: Output directory for checkpoints (defaults to `local_output`).
- `--tb-log-dir-path`: TensorBoard log directory.
- `--log-txt-path`, `--last-ckpt-path`: Managed automatically via `auto_resume`.

### Compilation
- `--vfast {0,1,2}`: Torch compile the VAE (`0` disables).
- `--tfast {0,1,2}`: Torch compile StyleVAR.

### Architecture / Init
- `--depth`: Transformer depth (default `20`).
- `--style-enc-dim`: Style encoder dimension (default `512`).
- `--ini`: Global init scale (`-1` lets the model auto-compute).
- `--hd`: Head layer multiplier.
- `--aln`, `--alng`: AdaLN init multipliers.
- `--saln`: Use shared AdaLN layers.
- `--anorm`: Use L2-normalized attention.
- `--fuse`: Enable fused ops / flash-attn / xFormers (set `0` to disable).

### Precision / Optimizer
- `--fp16 {0,1,2}`: 0=fp32, 1=fp16, 2=bf16.
- `--tblr`: Base learning rate used to derive `--tlr`.
- `--tlr`: Override learning rate directly.
- `--twd`: Initial weight decay.
- `--twde`: Final weight decay.
- `--tclip`: Gradient clipping threshold (`<=0` disables).
- `--ls`: Label smoothing.
- `--opt {adamw,adam}`: Optimizer choice (both map to AdamW).
- `--afuse`: Enable fused AdamW kernels (requires `--fuse 1` and compiled support).

### Batch & Schedule
- `--bs`: Target global batch size.
- `--batch-size`: Per-GPU batch (auto-derived, rarely set manually).
- `--glb-batch-size`: Global batch (auto).
- `--ac`: Gradient accumulation steps.
- `--ep`: Total epochs.
- `--wp`: Fraction of training for LR warmup.
- `--wp0`: LR ratio at warmup start.
- `--wpe`: LR ratio at training end.
- `--sche`: LR schedule (`lin1`, `lin0`, etc.).

### Data / Augmentation
- `--pn`: Patch grid definition (`1_2_3_4_5_6_8_10_13_16` by default).
- `--patch-size`: Patch resolution (default `16`).
- `--mid-reso`: Resize multiplier before cropping (default `1.125`).
- `--hflip`: Random horizontal flip toggle.
- `--workers`: Dataloader worker count (default `6`).
- `--data-load-reso`: Auto-computed from `pn * patch_size`; override only for experimentation.

### Progressive Training
- `--pg`: Portion of training that uses progressive stages (0 disables).
- `--pg0`: Initial progressive stage index.
- `--pgwp`: Warmup epochs per progressive stage.

### Misc / Environment
- `--exp-name`: Experiment tag for logging.
- `--seed`: Base random seed (also seeds workers if provided).
- `--same-seed-for-all-ranks`: Use one seed across ranks for data sampling.
- `--tf32`: Allow TensorFloat32 matmuls when available.
- `--device`: Target device (auto-set).
- `--local-debug`: Enables synthetic data/testing shortcuts.
- `--dbg-nan`: Optional NaN debugging.

## Workflow Tips
1. **Checkpoints**: Place `vae_ch160v4096z32.pth` and `var_d20.pth` under `origin_checkpoints/` or pass explicit paths via `--vae-ckpt-path` / `--vanilla-ckpt-path`.
2. **Dataset Smoke Test**: Run `python train_style_var/test_dataset.py --data-root ./dataset` to ensure target/style/content triplets load correctly.
3. **Disable fused kernels** unless flash-attention/xFormers are built in your environment (`--fuse 0 --afuse 0`).
4. **Monitor outputs**: checkpoints live in `local_output/<exp_name>/` and the best model is automatically mirrored to `ar-ckpt-best.pth` whenever validation tail loss improves.

Armed with these launch templates and argument descriptions, you can tailor `train_style_var/fine_tune.py` to any training environment.***
