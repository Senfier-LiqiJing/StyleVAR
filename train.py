import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import dist
from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume
from utils.data import normalize_01_into_pm1


_valid_cache = {}


def _build_valid_transform(reso: int, mid_reso_mult: float):
    mid_reso = round(reso * mid_reso_mult)
    return transforms.Compose([
        transforms.Resize(mid_reso, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(reso),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])


def _get_cached_valid_imgs(args: arg_util.Args):
    device = args.device
    cache = _valid_cache.get('imgs')
    if cache is not None and cache['style'].device == device and cache['content'].device == device:
        return cache['style'], cache['content']
    tfm = _build_valid_transform(args.data_load_reso, args.mid_reso)
    if not os.path.isfile(args.valid_style_path) or not os.path.isfile(args.valid_content_path):
        raise FileNotFoundError(f'valid style/content images not found: {args.valid_style_path}, {args.valid_content_path}')
    style = tfm(Image.open(args.valid_style_path).convert('RGB')).unsqueeze(0).to(device)
    content = tfm(Image.open(args.valid_content_path).convert('RGB')).unsqueeze(0).to(device)
    _valid_cache['imgs'] = {'style': style, 'content': content}
    return style, content


@torch.no_grad()
def _log_valid_samples_if_needed(args: arg_util.Args, trainer, tb_lg, global_step: int):
    if args.valid_every <= 0:
        return
    if (global_step + 1) % args.valid_every != 0:
        return
    if not dist.is_master():
        return
    try:
        print(f'[valid] step={global_step+1} using style={args.valid_style_path}, content={args.valid_content_path}')
        style_img, content_img = _get_cached_valid_imgs(args)
        was_training = trainer.var_wo_ddp.training
        trainer.var_wo_ddp.eval()
        out = trainer.var_wo_ddp.autoregressive_infer_cfg(
            B=1,
            style_img=style_img,
            content_img=content_img,
            g_seed=args.seed,
            cfg=1.5,
            top_k=0,
            top_p=0.0,
            more_smooth=False,
        )
        if was_training:
            trainer.var_wo_ddp.train()
        style_vis = style_img[0].detach().cpu().mul(0.5).add_(0.5).clamp(0, 1)
        content_vis = content_img[0].detach().cpu().mul(0.5).add_(0.5).clamp(0, 1)
        out_vis = out[0].detach().cpu().clamp(0, 1)
        # Ensure step is set before logging images
        tb_lg.set_step(global_step)
        # Log images with explicit step to ensure they're recorded correctly
        try:
            tb_lg.log_image('valid/style', style_vis, step=global_step)
            tb_lg.log_image('valid/content', content_vis, step=global_step)
            tb_lg.log_image('valid/output', out_vis, step=global_step)
            print(f'[valid] images logged to wandb/tensorboard at step {global_step}')
        except Exception as e:
            print(f'[valid] failed to log images to wandb/tensorboard: {e}')
            import traceback
            print(f'[valid] traceback: {traceback.format_exc()}')
        save_dir = os.path.join(args.local_out_dir_path, 'valid_outputs')
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f'step{global_step+1}.png')
        save_image(out_vis, out_path)
        print(f'[valid] output saved to {out_path}')
    except Exception as exc:
        print(f'[valid inference] failed at step {global_step}: {exc}')


def build_everything(args: arg_util.Args):
    # resume
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    # create tensorboard logger
    tb_lg: misc.LoggerGroup
    loggers = []
    with_tb_lg = dist.is_master()
    wandb_logger = None
    if with_tb_lg:
        if args.use_wandb:
            try:
                import wandb
                # Set environment variable to avoid TTY issues
                os.environ['WANDB_CONSOLE'] = 'wrap'
                # Temporarily restore original stdout/stderr for wandb init
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                if isinstance(sys.stdout, misc.SyncPrint):
                    sys.stdout = sys.stdout.terminal_stream
                if isinstance(sys.stderr, misc.SyncPrint):
                    sys.stderr = sys.stderr.terminal_stream
                
                try:
                    wandb_run = wandb.init(
                        project=args.wandb_project,
                        name=args.wandb_run_name,
                        config=args.state_dict(),
                        dir=args.local_out_dir_path,
                        settings=wandb.Settings(
                            mode='online',
                            console='wrap',  # Use wrap mode to avoid TTY issues
                            _disable_stats=False,
                        ),
                    )
                    wandb_logger = misc.WandbLogger(wandb_run)
                    loggers.append(wandb_logger)
                    print('[wandb] online logging enabled; tensorboard disabled')
                finally:
                    # Restore SyncPrint
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
            except Exception as e:
                import traceback
                print(f'[warn] failed to init wandb: {e}')
                print(f'[warn] wandb traceback: {traceback.format_exc()}')
        else:
            os.makedirs(args.tb_log_dir_path, exist_ok=True)
            loggers.append(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'))
    tb_lg = misc.DistLogger(misc.LoggerGroup(*loggers), verbose=with_tb_lg)
    tb_lg.flush()
    dist.barrier()
    
    # log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    # build data
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        num_classes, dataset_train, dataset_val = build_dataset(
            args.data_path, final_reso=args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso,
        )
        types = str((type(dataset_train).__name__, type(dataset_val).__name__))
        
        ld_val = DataLoader(
            dataset_val, num_workers=0, pin_memory=True,
            batch_size=round(args.batch_size*1.5), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
            shuffle=False, drop_last=False,
        )
        del dataset_val
        
        ld_train = DataLoader(
            dataset=dataset_train, num_workers=args.workers, pin_memory=True,
            generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
            ),
        )
        del dataset_train
        
        [print(line) for line in auto_resume_info]
        print(f'[dataloader multi processing] ...', end='', flush=True)
        stt = time.time()
        iters_train = len(ld_train)
        ld_train = iter(ld_train)
        # noinspection PyArgumentList
        print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
        print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
    
    else:
        num_classes = 1000
        ld_val = ld_train = None
        iters_train = 10
    
    # build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import StyleVAR, VQVAE
    from trainer import StyleVARTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params

    vae_local = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=args.patch_nums).to(dist.get_device())
    if args.vae_ckpt_path:
        vae_state = torch.load(args.vae_ckpt_path, map_location='cpu')
        vae_local.load_state_dict(vae_state, strict=True)
    [p.requires_grad_(False) for p in vae_local.parameters()]
    vae_local.eval()

    heads = args.depth
    width = args.depth * 64
    dpr = 0.1 * args.depth / 24
    var_kwargs = dict(
        num_classes=num_classes,
        depth=args.depth,
        embed_dim=width,
        num_heads=heads,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=dpr,
        norm_eps=1e-6,
        shared_aln=args.saln,
        cond_drop_rate=0.1,
        style_enc_dim=args.style_enc_dim,
        attn_l2_norm=args.anorm,
        patch_nums=args.patch_nums,
        flash_if_available=args.fuse,
        fused_if_available=args.fuse,
        alpha_nums=args.alpha_nums,
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    var_wo_ddp, load_info = StyleVAR.from_pretrained(
        vae_local=vae_local,
        pretrained_path=args.vanilla_ckpt_path,
        map_location='cpu',
        zero_unmatched=bool(args.zero_unmatched),
        freeze_resnet=args.freeze_resnet,
        freeze_backbone=args.freeze_backbone,
        **var_kwargs,
    )
    var_wo_ddp = var_wo_ddp.to(dist.get_device())

    var_wo_ddp: StyleVAR = args.compile_model(var_wo_ddp, args.tfast)
    var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    
    print(f'[INIT] StyleVAR model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('StyleVAR', var_wo_ddp),)]) + '\n\n')
    print(f'[INIT][pretrained] loaded={len(load_info.get("loaded", []))}, zeroed={len(load_info.get("zeroed", []))}')
    
    # build optimizer
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    
    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    # build trainer
    trainer = StyleVARTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var,
        var_opt=var_optim, label_smooth=args.ls,
    )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    del vae_local, var_wo_ddp, var, var_optim
    
    if args.local_debug:
        rng = torch.Generator('cpu')
        rng.manual_seed(0)
        B = 4
        target = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        style = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        content = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        
        me = misc.MetricLogger(delimiter='  ')
        trainer.train_step(
            it=0, g_it=0, stepping=True, metric_lg=me, tb_lg=tb_lg,
            target_B3HW=target, style_B3HW=style, content_B3HW=content, prog_si=args.pg0, prog_wp_it=20,
        )
        trainer.load_state_dict(trainer.state_dict())
        trainer.train_step(
            it=99, g_it=599, stepping=True, metric_lg=me, tb_lg=tb_lg,
            target_B3HW=target, style_B3HW=style, content_B3HW=content, prog_si=-1, prog_wp_it=20,
        )
        print({k: meter.global_avg for k, meter in me.meters.items()})
        
        args.dump_log(); tb_lg.flush(); tb_lg.close()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
        exit(0)
    
    dist.barrier()
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val
    )


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    
    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val
    ) = build_everything(args)
    
    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1
    
    L_mean, L_tail = -1, -1
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        tb_lg.set_step(ep * iters_train)
        
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, args, tb_lg, ld_train, iters_train, trainer
        )
        
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail)
        is_val_and_also_saving = (ep + 1) % 10 == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)
            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail, val_acc_tail)
            AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean, vacc_tail=val_acc_tail)
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
            print(f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s')
            
            if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, local_out_ckpt)
                if best_updated:
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
            dist.barrier()
        
        print(    f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        tb_lg.update(head='AR_ep_loss', step=ep+1, **AR_ep_loss)
        tb_lg.update(head='AR_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
        args.dump_log(); tb_lg.flush()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); tb_lg.flush(); tb_lg.close()
    dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.LoggerGroup, ld_or_itrt, iters_train: int, trainer):
    # import heavy packages after Dataloader object creation
    from trainer import VARTrainer
    from utils.lr_control import lr_wd_annealing
    trainer: VARTrainer
    
    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    pbar = tqdm(
        range(start_it, iters_train),
        total=iters_train,
        initial=start_it,
        disable=not dist.is_local_master(),
        dynamic_ncols=True,
        leave=False,
        desc=f'{header}',
    )
    for it in range(start_it, iters_train):
        batch = next(ld_or_itrt)
        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        target, style, content = batch
        target = target.to(args.device, non_blocking=True)
        style = style.to(args.device, non_blocking=True)
        content = content.to(args.device, non_blocking=True)
        
        args.cur_it = f'{it+1}/{iters_train}'
        
        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        
        if args.pg: # default: args.pg == 0.0, means no progressive training, won't get into this
            if g_it <= wp_it: prog_si = args.pg0
            elif g_it >= max_it*args.pg: prog_si = len(args.patch_nums) - 1
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(max((g_it - wp_it) / (max_it*args.pg - wp_it), 0), 1) # from 0 to 1
                prog_si = args.pg0 + round(progress * delta)    # from args.pg0 to len(args.patch_nums)-1
        else:
            prog_si = -1
        
        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)
        
        grad_norm, scale_log2 = trainer.train_step(
            it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
            target_B3HW=target, style_B3HW=style, content_B3HW=content, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        )

        if args.save_every > 0 and (g_it + 1) % args.save_every == 0:
            if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, f'ar-ckpt-step{g_it+1}.pth')
                print(f'[saving ckpt @ step {g_it+1}] ...', end='', flush=True)
                torch.save({
                    'epoch':    ep,
                    'iter':     it+1,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, local_out_ckpt)
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
            dist.barrier()

        me.update(tlr=max_tlr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
        tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)
        
        if args.tclip > 0:
            tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
            tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)
        
        # Log system stats to wandb periodically (every 10 iterations to avoid overhead)
        if args.use_wandb and (g_it == 0 or (g_it + 1) % 10 == 0):
            tb_lg.log_system_stats(step=g_it)
        
        # Flush wandb periodically to ensure data is sent
        if args.use_wandb and (g_it == 0 or (g_it + 1) % 50 == 0):
            tb_lg.flush()
        
        _log_valid_samples_if_needed(args, trainer, tb_lg, g_it)
    
        if dist.is_local_master():
            pbar.set_postfix({
                'Lm': f"{me.meters['Lm'].median:.3f}" if 'Lm' in me.meters else '-',
                'Acc': f"{me.meters['Accm'].median:.2f}" if 'Accm' in me.meters else '-',
                'tlr': f"{max_tlr:.2g}",
            })
            pbar.update(1)
    if dist.is_local_master():
        pbar.close()
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
