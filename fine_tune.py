import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import dist
from utils import arg_util, misc
from utils.data import build_dataset  # Assumes this function now returns (target, style, content) tuples
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume

# --- MODIFIED IMPORTS ---
# Import your new StyleVAR model and the corresponding builder
from models import StyleVAR, VQVAE, build_vae_stylevar 
# Import your new StyleVARTrainer from fine_tuner.py
from fine_tuner import StyleVARTrainer
# --- END MODIFIED IMPORTS ---

def build_everything(args: arg_util.Args):
    # resume
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    # create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'), verbose=True)
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()

    # ---- wandb ----
    wandb_run = None
    if dist.is_master() and args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=args.exp_name,
                id=args.wandb_run_id or None,
                config={k: v for k, v in args.state_dict().items()
                        if not k.startswith('cur_') and k not in
                        {'device', 'remain_time', 'finish_time'}},
                resume='allow',
            )
            print(f'[wandb] Initialized: {wandb_run.url}')
        except Exception as e:
            print(f'[wandb] Failed to init: {e}, continuing without wandb')
            wandb_run = None

    # log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
   
    # build data
    curriculum_dataset = None   # will be set if curriculum mixed training
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        if args.new_data_path or args.new_data_tar_dir:
            # ---- Curriculum mixed fine-tuning ----
            from utils.data import build_curriculum_dataset
            _, dataset_train, dataset_val = build_curriculum_dataset(
                old_data_path=args.data_path,
                new_data_path=args.new_data_path,
                new_data_tar_dir=args.new_data_tar_dir,
                final_reso=args.data_load_reso,
                start_ratio=args.curriculum_start,
                hflip=args.hflip, mid_reso=args.mid_reso,
            )
            curriculum_dataset = dataset_train  # keep ref for ratio updates
        else:
            # ---- Original OmniStyle-only training ----
            _, dataset_train, dataset_val = build_dataset(
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
        if curriculum_dataset is None:
            del dataset_train
        
        [print(line) for line in auto_resume_info]
        print(f'[dataloader multi processing] ...', end='', flush=True)
        stt = time.time()
        iters_train = len(ld_train)
        ld_train = iter(ld_train)
        # noinspection PyArgumentList
        print(f'         [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
        print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
   
    else:
        # num_classes = 1000 # No longer needed for StyleVAR
        ld_val = ld_train = None
        iters_train = 10
   
    # build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    # from models import StyleVAR, VQVAE, build_vae_stylevar # Already imported at top
    # from fine_tuner import StyleVARTrainer # Already imported at top
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params
   
    vae_local, var_wo_ddp = build_vae_stylevar(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,       # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums,
        # num_classes=num_classes, # No longer needed for StyleVAR
        depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
        # Add any new args for StyleVAR here, e.g., style_enc_dim
        style_enc_dim=512
    )
   
    # VAE checkpoint path
    vae_ckpt = os.path.join(os.path.dirname(__file__), 'ckpt', 'vae_ch160v4096z32.pth')
    if not os.path.exists(vae_ckpt):
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt}")
    dist.barrier()
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
   
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    # --- MODIFIED TYPE HINT ---
    var_wo_ddp: StyleVAR = args.compile_model(var_wo_ddp, args.tfast)

    # Alpha nums from command line
    if hasattr(args, 'alpha_nums') and args.alpha_nums:
        alpha_tuple = tuple(float(x) for x in args.alpha_nums.replace('-', '_').split('_'))
        assert len(alpha_tuple) == len(var_wo_ddp.alpha_nums), \
            f"alpha_nums length {len(alpha_tuple)} != patch_nums length {len(var_wo_ddp.alpha_nums)}"
        var_wo_ddp.alpha_nums = alpha_tuple
        if dist.is_local_master():
            print(f"[Config] Alpha nums: {var_wo_ddp.alpha_nums}")

    # Alpha jitter for blended cross-attention robustness
    if hasattr(args, 'alpha_jitter') and args.alpha_jitter > 0:
        var_wo_ddp.alpha_jitter = args.alpha_jitter
        if dist.is_local_master():
            print(f"[Config] Alpha jitter enabled: ±{args.alpha_jitter}")

    # ================================================================
    # [OPTIONAL] High-rank LoRA — uncomment below if OOM on full-param
    # Requires:  pip install peft
    # ================================================================
    # from peft import LoraConfig, get_peft_model
    # lora_config = LoraConfig(
    #     r=64,                          # high rank for visual generation (try 64 or 128)
    #     lora_alpha=128,                # scaling = alpha / r
    #     target_modules=[
    #         "mat_qkv_guide",           # cross-attn guide QKV
    #         "mat_qkv_target",          # cross-attn target QKV
    #         "proj",                    # attn output projection
    #         # "fc1", "fc2",            # uncomment to also adapt FFN layers
    #     ],
    #     lora_dropout=0.05,
    #     bias="none",
    # )
    # var_wo_ddp = get_peft_model(var_wo_ddp, lora_config)
    # var_wo_ddp.print_trainable_parameters()
    # ================================================================

    is_resuming = (trainer_state is not None) and (len(trainer_state) > 0)

    if is_resuming:
        if dist.is_local_master():
            print(f"[Config] 🚀 Detect resume checkpoint (Epoch {start_ep}). SKIPPING original VAR weight loading.")

    else:
        # --- Not Resuming: Initialize Weights ---
        clean_ckpt_path = args.clean_ckpt_path
        vanilla_ckpt_path = args.vanilla_ckpt_path
        
        # 1. Try to load the Clean Checkpoint (Exact Match)
        if os.path.exists(clean_ckpt_path):
            if dist.is_local_master():
                print(f"[Config] Found CLEAN fine-tuned checkpoint: {clean_ckpt_path}")
                print("[Config] Loading this as initialization (Fresh Start)...")
            
            try:
                # Load directly as the structure should match
                state_dict = torch.load(clean_ckpt_path, map_location='cpu')
                
                # Handle potential 'model' or 'trainer' wrappers in the saved file
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'trainer' in state_dict and 'var_wo_ddp' in state_dict['trainer']:
                    state_dict = state_dict['trainer']['var_wo_ddp']

                missing_keys, unexpected_keys = var_wo_ddp.load_state_dict(state_dict, strict=False)
                
                if dist.is_local_master():
                    print(f"Loaded clean checkpoint. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                    
            except Exception as e:
                print(f"Error loading clean checkpoint: {e}")
                raise e

        # 2. Fallback to Vanilla VAR Checkpoint (Complex Weight Transfer)
        else:
            if dist.is_local_master():
                print(f"[Config] No clean checkpoint found. Loading ORIGINAL VAR: {vanilla_ckpt_path}")
            
            try:
                full_ckpt = torch.load(vanilla_ckpt_path, map_location='cpu')
            except Exception as e:
                raise FileNotFoundError(f"Failed to load vanilla checkpoint: {e}")

            # Extract weights from vanilla checkpoint
            if 'trainer' in full_ckpt and 'var_wo_ddp' in full_ckpt['trainer']:
                var_state_dict = full_ckpt['trainer']['var_wo_ddp']
                if dist.is_local_master(): print("Found model weights inside 'trainer.var_wo_ddp'.")
            elif 'model' in full_ckpt:
                var_state_dict = full_ckpt['model']
            else:
                # Assume the checkpoint itself is the state_dict
                var_state_dict = full_ckpt

            # Perform Weight Transfer
            if dist.is_local_master(): print("Starting weight transfer logic...")
            
            new_state_dict = OrderedDict()
            style_model_keys = var_wo_ddp.state_dict().keys()

            for k, v in var_state_dict.items():
                # Map Attention Weights (QKV -> Guide/Target)
                if 'attn.mat_qkv.weight' in k:
                    guide_key = k.replace('attn.mat_qkv.weight', 'attn.mat_qkv_guide.weight')
                    target_key = k.replace('attn.mat_qkv.weight', 'attn.mat_qkv_target.weight')
                    if guide_key in style_model_keys: new_state_dict[guide_key] = v
                    if target_key in style_model_keys: new_state_dict[target_key] = v
                
                # Map Biases
                elif 'attn.q_bias' in k:
                    guide_key = k.replace('attn.q_bias', 'attn.q_bias_guide')
                    target_key = k.replace('attn.q_bias', 'attn.q_bias_target')
                    if guide_key in style_model_keys: new_state_dict[guide_key] = v
                    if target_key in style_model_keys: new_state_dict[target_key] = v
                elif 'attn.v_bias' in k:
                    guide_key = k.replace('attn.v_bias', 'attn.v_bias_guide')
                    target_key = k.replace('attn.v_bias', 'attn.v_bias_target')
                    if guide_key in style_model_keys: new_state_dict[guide_key] = v
                    if target_key in style_model_keys: new_state_dict[target_key] = v
                
                # Direct Copy for matching keys
                elif k in style_model_keys:
                    new_state_dict[k] = v
                else:
                    # Key exists in Vanilla but not in StyleVAR (e.g. unused class embeddings)
                    pass

            # Load the mapped weights
            missing_keys, unexpected_keys = var_wo_ddp.load_state_dict(new_state_dict, strict=False)

            if dist.is_local_master():
                print("\n--- Weight Loading Summary (Vanilla Transfer) ---")
                print(f"Successfully loaded {len(new_state_dict)} tensors.")
                print("\nMissing keys (Expected and will be fine-tuned):")
                for key in sorted(missing_keys): print(f"  {key}")
                print("\nUnexpected keys (Should be empty):")
                for key in sorted(unexpected_keys): print(f"  {key}")
                print("\nModel loading complete.")

    var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    print(f'[INIT] VAR model = {var_wo_ddp}\n\n')
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
   
    # --- MODIFIED DEBUG BLOCK ---
    if args.local_debug:
        rng = torch.Generator('cpu')
        rng.manual_seed(0)
        B = 4
        # Create dummy data for (target, style, content)
        inp = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        style = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        content = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        
        me = misc.MetricLogger(delimiter='  ')
        # Call train_step with the new signature
        trainer.train_step(
            it=0, g_it=0, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, style_B3HW=style, content_B3HW=content,
            prog_si=args.pg0, prog_wp_it=20,
        )
        trainer.load_state_dict(trainer.state_dict())
        trainer.train_step(
            it=99, g_it=599, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, style_B3HW=style, content_B3HW=content,
            prog_si=-1, prog_wp_it=20,
        )
        print({k: meter.global_avg for k, meter in me.meters.items()})
        
        args.dump_log(); tb_lg.flush(); tb_lg.close()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
        exit(0)
    # --- END MODIFIED DEBUG BLOCK ---
   
    dist.barrier()
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val,
        curriculum_dataset,             # None when not using curriculum
        wandb_run,
    )

def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)

    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val,
        curriculum_dataset,
        wandb_run,
    ) = build_everything(args)
    print("[INIT] Build Everything Ready.")

    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1

    L_mean, L_tail = -1, -1
    for ep in range(start_ep, args.ep):
        # ---- Curriculum ratio update (linear schedule) ----
        if curriculum_dataset is not None:
            progress = ep / max(args.ep - 1, 1)
            ratio = args.curriculum_start + \
                    (args.curriculum_end - args.curriculum_start) * progress
            curriculum_dataset.set_new_data_ratio(ratio)
            if dist.is_local_master():
                print(f'[Curriculum] Ep {ep}: new_data_ratio = {ratio:.3f}')

        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        tb_lg.set_step(ep * iters_train)
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, args, tb_lg, ld_train, iters_train, trainer,
            wandb_run=wandb_run,
        )
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail)
        is_val_and_also_saving = (ep + 1) % 1 == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)
            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail, val_acc_tail)
            AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean, vacc_tail=val_acc_tail)
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
            print(f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s')
            
            if dist.is_local_master():
                ckpt_state = {
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }
                # last (for auto_resume)
                misc.atomic_save(ckpt_state, os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth'))
                print(f'[ckpt] last saved (ep{ep+1})', flush=True, clean=True)
                # best (by val loss tail)
                if best_updated:
                    shutil.copy(os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth'),
                                os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth'))
                    print(f'[ckpt] best updated (vL_tail={val_loss_tail:.4f})', flush=True, clean=True)
            dist.barrier()
        
        print(    f'         [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        tb_lg.update(head='AR_ep_loss', step=ep+1, **AR_ep_loss)
        tb_lg.update(head='AR_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
        # ---- wandb epoch logging ----
        if wandb_run is not None:
            # Use iters_train-based step to stay monotonic with iter-level logs
            ep_step = (ep + 1) * iters_train
            wandb_log = {f'epoch/{k}': v for k, v in AR_ep_loss.items()}
            wandb_log['epoch/epoch'] = ep + 1
            wandb_log['epoch/hours'] = round(sec / 60 / 60, 2)
            wandb_log['epoch/lr'] = args.cur_lr
            if curriculum_dataset is not None:
                wandb_log['epoch/new_data_ratio'] = curriculum_dataset.new_data_ratio
            wandb_run.log(wandb_log, step=ep_step)
        args.dump_log(); tb_lg.flush()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},    Lm: {best_L_mean:.3f} ({L_mean}),    Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')

    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); tb_lg.flush(); tb_lg.close()
    if wandb_run is not None:
        wandb_run.finish()
    dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer, wandb_run=None):
    # import heavy packages after Dataloader object creation
    # --- MODIFIED IMPORT ---
    from fine_tuner import StyleVARTrainer
    # --- END MODIFIED IMPORT ---
    from utils.lr_control import lr_wd_annealing
    trainer: StyleVARTrainer
    
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
    
    # --- MODIFIED DATALOADER LOOP ---
    # The dataloader now yields (target, style, content) as 'inp', 'style', 'content'
    for it, (inp, style, content) in me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
    # --- END MODIFIED DATALOADER LOOP ---
        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        # --- MODIFIED TENSOR TO DEVICE ---
        inp = inp.to(args.device, non_blocking=True)
        style = style.to(args.device, non_blocking=True)
        content = content.to(args.device, non_blocking=True)
        # 'label' is no longer used
        # --- END MODIFIED TENSOR TO DEVICE ---
        
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
        
        # --- MODIFIED TRAINER CALL ---
        # Pass the new style and content tensors to the trainer
        grad_norm, scale_log2 = trainer.train_step(
            it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, style_B3HW=style, content_B3HW=content, 
            prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        )
        # --- END MODIFIED TRAINER CALL ---
        
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

        # ---- wandb iter logging (every optimizer step) ----
        if wandb_run is not None and stepping:
            wandb_run.log({
                'train/loss_mean': me.meters['Lm'].avg if 'Lm' in me.meters else None,
                'train/loss_tail': me.meters['Lt'].avg if 'Lt' in me.meters else None,
                'train/acc_mean': me.meters['Accm'].avg if 'Accm' in me.meters else None,
                'train/acc_tail': me.meters['Acct'].avg if 'Acct' in me.meters else None,
                'train/grad_norm': grad_norm,
                'train/lr': max_tlr,
                'train/wd': max_twd,
            }, step=g_it)

        # --- Rolling checkpoint every save_every iterations ---
        if args.save_every > 0 and (it + 1) % args.save_every == 0 and dist.is_local_master():
            roll_state = {
                'epoch':   ep,
                'iter':    it + 1,
                'trainer': trainer.state_dict(),
                'args':    args.state_dict(),
            }
            roll_path = misc.save_rolling_checkpoint(
                roll_state, args.local_out_dir_path,
                max_slots=args.max_rolling,
            )
            # also keep ar-ckpt-last.pth up to date
            misc.atomic_save(roll_state, os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth'))
            print(f'[rolling ckpt] ep{ep} it{it+1}/{iters_train} -> {os.path.basename(roll_path)}', flush=True, clean=True)

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