# StyleVAR 微调脚本使用指南（中文）

`train_style_var/fine_tune.py` 复用了主工程的 `utils.arg_util.Args` 参数系统，因而可以继承原始 VAR 训练脚本的所有命令行参数，这份文档将用中文介绍如何启动、如何关闭 flash-attention / xFormers、以及每个参数的含义。

## 启动示例

### 单卡 / 调试
```bash
python train_style_var/fine_tune.py \
  --data-path ./dataset \
  --origin-ckpt-dir ./origin_checkpoints \
  --local-out-dir-path ./local_output/stylevar_ft \
  --tb-log-dir-path ./local_output/stylevar_ft/tb \
  --ep 5 --bs 16 --workers 4 --fuse 0
```

### 多卡（示例：8 卡 torchrun）
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

> **关于 flash-attention/xFormers**：通过 `--fuse` 控制。若环境没有这些依赖，请设定 `--fuse 0`（同时建议 `--afuse 0`）。只有在确认 flash-attn/xFormers/fused kernel 已正确安装后才开启 `--fuse 1`。

## 参数总览

以下所有参数都可以传给 `train_style_var/fine_tune.py`，默认值见 `utils/arg_util.Args`。

### 路径相关
- `--data-path`：内容/风格/目标图片所在的根目录（默认 `/home/OmniStyle-150K`）。
- `--origin-ckpt-dir`：默认的 checkpoint 文件夹，内含 `vae_ch160v4096z32.pth` 与 `var_d20.pth`。
- `--clean-ckpt-path`：若已有 fine-tune 完成的 StyleVAR 权重，可通过此参数直接加载。
- `--vanilla-ckpt-path`：原始 VAR 模型 checkpoint。
- `--vae-ckpt-path`：预训练 tokenizer/VAE checkpoint。
- `--local-out-dir-path`：保存训练中 checkpoint 的目录（默认 `local_output`）。
- `--tb-log-dir-path`：TensorBoard 日志目录。
- `--log-txt-path` / `--last-ckpt-path`：由 `auto_resume` 自动维护。

### 编译选项
- `--vfast {0,1,2}`：控制是否对 VAE 做 torch.compile。
- `--tfast {0,1,2}`：控制是否对 StyleVAR 做 torch.compile。

### 模型与初始化
- `--depth`：Transformer 深度（默认 20）。
- `--style-enc-dim`：风格编码维度（默认 512）。
- `--ini`：整体初始化尺度（`-1` 表示自动）。
- `--hd`：输出头部缩放系数。
- `--aln`，`--alng`：AdaLN 初始化倍数。
- `--saln`：启用共享 AdaLN。
- `--anorm`：使用 L2 normalized attention。
- `--fuse`：是否启用 fused ops / flash-attn / xFormers（默认 0）。

### 精度与优化器
- `--fp16 {0,1,2}`：0=FP32，1=FP16，2=BF16。
- `--tblr`：基础学习率，用于推导 `--tlr`。
- `--tlr`：直接指定学习率（通常保留默认，根据 `tblr` 和 batch 自动算）。
- `--twd` / `--twde`：初始/最终 weight decay。
- `--tclip`：梯度裁剪阈值（`<=0` 表示不裁剪）。
- `--ls`：label smoothing。
- `--opt {adamw,adam}`：优化器类型（本质上均映射到 AdamW）。
- `--afuse`：开启 PyTorch fused AdamW，需要 `--fuse 1` 且环境支持。

### 批大小与学习率计划
- `--bs`：目标全局 batch size。
- `--batch-size` / `--glb-batch-size`：通常由脚本自动计算，不建议手动设置。
- `--ac`：梯度累积步数。
- `--ep`：训练 epoch 数。
- `--wp`：lr warmup 占比。
- `--wp0`：warmup 初始 lr 的比例。
- `--wpe`：训练结束时 lr 的比例。
- `--sche`：学习率 schedule（默认 `lin1`）。

### 数据与增强
- `--pn`：patch grid 定义（默认 `1_2_3_4_5_6_8_10_13_16`）。
- `--patch-size`：patch 尺寸（默认 16）。
- `--mid-reso`：裁剪前的 resize 倍数（默认 1.125）。
- `--hflip`：是否在训练集上使用随机水平翻转。
- `--workers`：DataLoader worker 数。
- `--data-load-reso`：自动推导的最大分辨率，通常无需手动赋值。

### Progressive Training
- `--pg`：>0 时启用 progressive training。
- `--pg0`：进度训练起始 stage。
- `--pgwp`：每个 stage 的 warmup 比例。

### 其他 / 环境
- `--exp-name`：实验名称，用于日志与输出目录。
- `--seed`：随机种子。
- `--same-seed-for-all-ranks`：分布式 sampler 是否共享同一 seed。
- `--tf32`：允许 TensorFloat32 运算。
- `--device`：目标设备（自动设置）。
- `--local-debug`：若开启会进入本地调试模式（使用伪造 batch）。
- `--dbg-nan`：是否启用 NaN 调试。

## 使用建议
1. **确保 checkpoint 准备齐全**：若不指定 `--vanilla-ckpt-path` / `--vae-ckpt-path`，脚本会自动在 `origin_checkpoints/` 中寻找 `var_d20.pth` 与 `vae_ch160v4096z32.pth`。
2. **优先跑数据检查**：`python train_style_var/test_dataset.py --data-root ./dataset` 用于确认 content/style/target 三元组是否能顺利读取。
3. **缺少 flash-attn/xFormers 时务必关闭 `--fuse`**（同时可把 `--afuse` 设为 `0`），防止运行期报错。
4. **关注输出目录**：`local_output/<exp_name>/` 中会保存最新 checkpoint，若 validation tail loss 下降，还会同步到 `ar-ckpt-best.pth`。

有了以上说明就可以根据自己的算力环境调整参数，顺利运行 `train_style_var/fine_tune.py`，开始 StyleVAR 的风格迁移微调了。
