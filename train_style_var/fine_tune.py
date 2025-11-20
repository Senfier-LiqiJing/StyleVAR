import gc
import os
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from train_style_var.fine_tuner import StyleVARTrainer
from utils import arg_util, misc
from utils.amp_sc import AmpOptimizer
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.lr_control import filter_params, lr_wd_annealing
from utils.misc import auto_resume
from models import StyleVAR, VQVAE, build_vae_stylevar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGIN_DIR = PROJECT_ROOT / "origin_checkpoints"


def _extract_cli_value(flag: str):
    prefix = f"{flag}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    for idx, arg in enumerate(sys.argv[:-1]):
        if arg == flag:
            return sys.argv[idx + 1]
    return None


CLI_BATCH_OVERRIDES = {
    "bs": _extract_cli_value("--bs"),
    "batch_size": _extract_cli_value("--batch_size"),
    "glb_batch_size": _extract_cli_value("--glb_batch_size"),
}


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _resolve_ckpt(path_candidate: str, fallback_name: str) -> Path:
    candidates = []
    if path_candidate:
        candidates.append(Path(path_candidate))
    candidates.append(DEFAULT_ORIGIN_DIR / fallback_name)
    for path in candidates:
        if path and path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Checkpoint for {fallback_name} not found in {candidates}")


def _load_pretrained_vae(vae: VQVAE, args) -> None:
    vae_ckpt = _resolve_ckpt(getattr(args, "vae_ckpt_path", ""), "vae_ch160v4096z32.pth")
    state = torch.load(vae_ckpt, map_location="cpu")
    vae.load_state_dict(state, strict=True)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)  # === FROZEN: keep tokenizer (VAE) fixed during StyleVAR fine-tuning ===
    print(f"[Init] Loaded and froze VAE from {vae_ckpt}")


def _extract_var_state(raw_state: dict) -> dict:
    if "trainer" in raw_state and "var_wo_ddp" in raw_state["trainer"]:
        return raw_state["trainer"]["var_wo_ddp"]
    if "model" in raw_state:
        return raw_state["model"]
    return raw_state


def _transfer_var_weights(var_model: StyleVAR, args) -> None:
    vanilla_ckpt = _resolve_ckpt(getattr(args, "vanilla_ckpt_path", ""), "var_d20.pth")
    raw = torch.load(vanilla_ckpt, map_location="cpu")
    source = _extract_var_state(raw)
    target_keys = var_model.state_dict().keys()
    mapped = OrderedDict()

    for key, value in source.items():
        if "attn.mat_qkv.weight" in key:
            guide = key.replace("attn.mat_qkv.weight", "attn.mat_qkv_guide.weight")
            target = key.replace("attn.mat_qkv.weight", "attn.mat_qkv_target.weight")
            if guide in target_keys:
                mapped[guide] = value  # === COPIED: guide attention inherits vanilla qkv ===
            if target in target_keys:
                mapped[target] = value  # === COPIED: target attention inherits vanilla qkv ===
            continue
        if "attn.q_bias" in key:
            guide = key.replace("attn.q_bias", "attn.q_bias_guide")
            target = key.replace("attn.q_bias", "attn.q_bias_target")
            if guide in target_keys:
                mapped[guide] = value  # === COPIED: guide q-bias reused ===
            if target in target_keys:
                mapped[target] = value  # === COPIED: target q-bias reused ===
            continue
        if "attn.v_bias" in key:
            guide = key.replace("attn.v_bias", "attn.v_bias_guide")
            target = key.replace("attn.v_bias", "attn.v_bias_target")
            if guide in target_keys:
                mapped[guide] = value  # === COPIED: guide v-bias reused ===
            if target in target_keys:
                mapped[target] = value  # === COPIED: target v-bias reused ===
            continue
        if key in target_keys:
            mapped[key] = value  # === COPIED: shared structural weights directly loaded ===

    missing, unexpected = var_model.load_state_dict(mapped, strict=False)
    print(f"[Init] Loaded StyleVAR from {vanilla_ckpt}")
    print(f"[Init] Missing keys: {len(missing)} | Unexpected: {len(unexpected)}")


def _apply_batch_overrides(args):
    world = max(1, dist.get_world_size())

    override_bs = CLI_BATCH_OVERRIDES.get("bs")
    override_local = CLI_BATCH_OVERRIDES.get("batch_size")
    override_global = CLI_BATCH_OVERRIDES.get("glb_batch_size")

    if override_local is not None:
        args.batch_size = max(1, int(override_local))
    if override_bs is not None:
        args.bs = max(1, int(override_bs))
        if override_global is None:
            override_global = args.bs
    if override_global is not None:
        args.glb_batch_size = max(world, int(override_global))

    if args.batch_size <= 0:
        approx = max(1, args.bs or args.glb_batch_size or 1)
        args.batch_size = max(1, approx // world)

    args.glb_batch_size = max(args.glb_batch_size, args.batch_size * world)
    args.bs = args.glb_batch_size
    args.workers = min(max(0, args.workers), args.batch_size)
    if args.glb_batch_size == 0:
        args.glb_batch_size = args.batch_size * world


def build_dataloaders(args, start_ep, start_it):
    num_classes, train_set, val_set = build_dataset(
        data_path=args.data_path,
        final_reso=args.data_load_reso,
        hflip=args.hflip,
        mid_reso=args.mid_reso,
    )
    if args.local_debug:
        dummy = torch.rand(args.batch_size, 3, args.data_load_reso, args.data_load_reso)
        sample = (dummy.clone(), dummy.clone(), dummy.clone())
        ld_train = iter([sample for _ in range(10)])
        ld_val = [sample]
        return num_classes, 10, ld_train, ld_val

    print("[Data] Building loaders with triplet dataset...")
    ld_val = DataLoader(
        val_set,
        num_workers=max(1, args.workers // 2),
        pin_memory=True,
        batch_size=max(1, round(args.batch_size * 1.5)),
        sampler=EvalDistributedSampler(val_set, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False,
        drop_last=False,
    )

    ld_train = DataLoader(
        dataset=train_set,
        num_workers=args.workers,
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(),
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(train_set),
            glb_batch_size=args.glb_batch_size,
            same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True,
            fill_last=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            start_ep=start_ep,
            start_it=start_it,
        ),
    )
    iters_train = len(ld_train)
    print(f"[Data] train={len(train_set)} val={len(val_set)} iters_train={iters_train}")
    return num_classes, iters_train, iter(ld_train), ld_val


def prepare_trainer(args, trainer_state):
    vae_local, var_wo_ddp = build_vae_stylevar(
        device=dist.get_device(),
        patch_nums=args.patch_nums,
        depth=args.depth,
        shared_aln=args.saln,
        attn_l2_norm=args.anorm,
        flash_if_available=False,  # === DISABLED: avoid flash-attn dependency ===
        fused_if_available=False,  # === DISABLED: avoid xFormers dependency ===
        init_adaln=args.aln,
        init_adaln_gamma=args.alng,
        init_head=args.hd,
        init_std=args.ini,
        style_enc_dim=getattr(args, "style_enc_dim", 512),
    )

    _load_pretrained_vae(vae_local, args)
    vae_local = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: StyleVAR = args.compile_model(var_wo_ddp, args.tfast)

    is_resuming = trainer_state is not None and len(trainer_state) > 0
    if not is_resuming:
        clean_ckpt = getattr(args, "clean_ckpt_path", "")
        if clean_ckpt and os.path.exists(clean_ckpt):
            state = torch.load(clean_ckpt, map_location="cpu")
            var_state = _extract_var_state(state)
            var_wo_ddp.load_state_dict(var_state, strict=False)
            print(f"[Init] Loaded clean fine-tuned weights from {clean_ckpt}")
        else:
            _transfer_var_weights(var_wo_ddp, args)
    else:
        print("[Init] Resume detected - skip weight init.")

    ddp_cls = DDP if dist.initialized() else NullDDP
    var = ddp_cls(
        var_wo_ddp,
        device_ids=[dist.get_local_rank()],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    names, params, param_groups = filter_params(
        var_wo_ddp,
        nowd_keys={
            "cls_token",
            "start_token",
            "task_token",
            "cfg_uncond",
            "pos_embed",
            "pos_1LC",
            "pos_start",
            "start_pos",
            "lvl_embed",
            "gamma",
            "beta",
            "ada_gss",
            "moe_bias",
            "scale_mul",
        },
    )
    opt_type = args.opt.lower().strip()
    opt_cls = {
        "adam": torch.optim.AdamW,
        "adamw": torch.optim.AdamW,
    }[opt_type]
    optimizer = opt_cls(params=param_groups, lr=args.tlr, weight_decay=0, betas=(0.9, 0.95), fused=False)
    var_opt = AmpOptimizer(
        mixed_precision=args.fp16,
        optimizer=optimizer,
        names=names,
        paras=params,
        grad_clip=args.tclip,
        n_gradient_accumulation=args.ac,
    )

    trainer = StyleVARTrainer(
        device=args.device,
        patch_nums=args.patch_nums,
        resos=args.resos,
        vae_local=vae_local,
        var_wo_ddp=var_wo_ddp,
        var=var,
        var_opt=var_opt,
        label_smooth=args.ls,
    )
    if trainer_state:
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True)
    return trainer


def build_everything(args: arg_util.Args):
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, "ar-ckpt*.pth")
    os.makedirs(args.tb_log_dir_path, exist_ok=True)
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    tb_logger = misc.DistLogger(
        misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f"__{misc.time_str('%m%d_%H%M')}"),
        verbose=dist.is_master(),
    )
    tb_logger.flush()

    num_classes, iters_train, ld_train, ld_val = build_dataloaders(args, start_ep, start_it)
    trainer = prepare_trainer(args, trainer_state)
    return tb_logger, trainer, start_ep, start_it, iters_train, ld_train, ld_val


def train_one_epoch(ep: int, start_it: int, args, tb_logger, trainer, iters_train, ld_train):
    me = misc.MetricLogger(delimiter="  ")
    me.add_meter("tlr", misc.SmoothedValue(window_size=1, fmt="{value:.2e}"))
    me.add_meter("tnm", misc.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    [me.add_meter(x, misc.SmoothedValue(fmt="{median:.3f} ({global_avg:.3f})")) for x in ["Lm", "Lt"]]
    [me.add_meter(x, misc.SmoothedValue(fmt="{median:.2f} ({global_avg:.2f})")) for x in ["Accm", "Acct"]]

    g_it = ep * iters_train
    max_it = args.ep * iters_train
    wp_it = args.wp * iters_train

    for it, (target, style, content) in me.log_every(start_it, iters_train, ld_train, 5, f"[Ep {ep}/{args.ep}]"):
        if it < start_it:
            continue
        g_it = ep * iters_train + it

        target = target.to(args.device, non_blocking=True)
        style = style.to(args.device, non_blocking=True)
        content = content.to(args.device, non_blocking=True)

        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
            args.sche,
            trainer.var_opt.optimizer,
            args.tlr,
            args.twd,
            args.twde,
            g_it,
            wp_it,
            max_it,
            wp0=args.wp0,
            wpe=args.wpe,
        )
        tb_logger.set_step(g_it)
        tb_logger.update(head="AR_opt_lr/lr_min", sche_tlr=min_tlr)
        tb_logger.update(head="AR_opt_lr/lr_max", sche_tlr=max_tlr)
        tb_logger.update(head="AR_opt_wd/wd_min", sche_twd=min_twd)
        tb_logger.update(head="AR_opt_wd/wd_max", sche_twd=max_twd)

        if args.pg:
            prog_si = trainer.compute_prog_stage(g_it, wp_it, max_it, args)
        else:
            prog_si = -1
        stepping = (g_it + 1) % args.ac == 0

        grad_norm, scale_log2 = trainer.train_step(
            it=it,
            g_it=g_it,
            stepping=stepping,
            metric_lg=me,
            tb_lg=tb_logger,
            inp_B3HW=target,
            style_B3HW=style,
            content_B3HW=content,
            prog_si=prog_si,
            prog_wp_it=args.pgwp * iters_train,
        )
        me.update(tlr=max_tlr)

        tb_logger.update(head="AR_opt_grad/fp16", scale_log2=scale_log2)
        if args.tclip > 0:
            tb_logger.update(head="AR_opt_grad/grad", grad_norm=grad_norm)

    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(
        max_it - (g_it + 1) + (args.ep - ep) * 15
    )


def launch_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    _apply_batch_overrides(args)
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)

    tb_logger, trainer, start_ep, start_it, iters_train, ld_train, ld_val = build_everything(args)
    print("[Init] Training stack ready.")

    best = {"Lm": float("inf"), "Lt": float("inf"), "Accm": -1.0, "Acct": -1.0}
    best_val_tail = float("inf")
    start_time = time.time()
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, "sampler") and hasattr(ld_train.sampler, "set_epoch"):
            ld_train.sampler.set_epoch(ep)
        tb_logger.set_step(ep * iters_train)

        stats, timing = train_one_epoch(ep, start_it if ep == start_ep else 0, args, tb_logger, trainer, iters_train, ld_train)
        best["Lm"] = min(best["Lm"], stats["Lm"])
        if stats["Lt"] != -1:
            best["Lt"] = min(best["Lt"], stats["Lt"])
        best["Accm"] = max(best["Accm"], stats["Accm"])
        if stats["Acct"] != -1:
            best["Acct"] = max(best["Acct"], stats["Acct"])

        args.L_mean, args.L_tail = stats["Lm"], stats["Lt"]
        args.acc_mean, args.acc_tail = stats["Accm"], stats["Acct"]
        args.grad_norm = stats["tnm"]
        args.cur_ep = f"{ep+1}/{args.ep}"
        args.remain_time, args.finish_time = timing[1], timing[2]

        if (ep + 1) % 1 == 0 or (ep + 1) == args.ep:
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)
            print(
                f"[Val] ep={ep} tot={tot} Lm={val_loss_mean:.4f} Lt={val_loss_tail:.4f} "
                f"Accm={val_acc_mean:.2f} Acct={val_acc_tail:.2f} cost={cost:.2f}s"
            )
            best_updated = val_loss_tail < best_val_tail
            best_val_tail = min(best_val_tail, val_loss_tail)
            if dist.is_local_master():
                local_out = os.path.join(args.local_out_dir_path, "ar-ckpt-last.pth")
                torch.save(
                    {"epoch": ep + 1, "iter": 0, "trainer": trainer.state_dict(), "args": args.state_dict()},
                    local_out,
                )
                if best_updated:
                    shutil.copy(local_out, os.path.join(args.local_out_dir_path, "ar-ckpt-best.pth"))

        tb_logger.update(head="AR_ep_loss", step=ep + 1, **stats)
        tb_logger.update(head="AR_z_burnout", step=ep + 1, rest_hours=round(timing[0] / 3600, 2))
        tb_logger.flush()

    total_time = (time.time() - start_time) / 3600
    print(f"[Done] Training finished in {total_time:.2f}h best={best}")
    tb_logger.close()
    dist.barrier()


if __name__ == "__main__":
    try:
        launch_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
