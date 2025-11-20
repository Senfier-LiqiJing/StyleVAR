import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import StyleVAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class StyleVARTrainer:
    def __init__(
        self,
        device,
        patch_nums: Tuple[int, ...],
        resos: Tuple[int, ...],
        vae_local: VQVAE,
        var_wo_ddp: StyleVAR,
        var: DDP,
        var_opt: AmpOptimizer,
        label_smooth: float,
    ):
        self.var = var  # === TRAINABLE: StyleVAR stays unfrozen to learn new style guidance ===
        self.vae_local = vae_local
        self.quantize_local: VectorQuantizer2 = vae_local.quantize
        self.var_wo_ddp: StyleVAR = var_wo_ddp
        self.var_opt = var_opt

        self.vae_local.eval()
        for param in self.vae_local.parameters():
            param.requires_grad_(False)  # === FROZEN: tokenizer/quantizer remain fixed for consistent codes ===

        if hasattr(self.var_wo_ddp, "rng"):
            del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction="none")
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction="mean")
        self.patch_nums = patch_nums
        self.resos = resos
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L

        self.begin_ends = []
        cursor = 0
        for pn in patch_nums:
            self.begin_ends.append((cursor, cursor + pn * pn))
            cursor += pn * pn

        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True

    def compute_prog_stage(self, g_it: int, wp_it: int, max_it: int, args) -> int:
        if not args.pg:
            return -1
        if g_it <= wp_it:
            prog_si = args.pg0
        elif g_it >= max_it * args.pg:
            prog_si = len(self.patch_nums) - 1
        else:
            delta = len(self.patch_nums) - 1 - args.pg0
            progress = min(max((g_it - wp_it) / (max_it * args.pg - wp_it), 0), 1)
            prog_si = args.pg0 + round(progress * delta)
        return prog_si

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        start = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for inp_B3HW, style_B3HW, content_B3HW in ld_val:
            B = inp_B3HW.shape[0]
            V = self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True).float()
            style_B3HW = style_B3HW.to(dist.get_device(), non_blocking=True).float()
            content_B3HW = content_B3HW.to(dist.get_device(), non_blocking=True).float()

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

            ms_style_idx = self.vae_local.img_to_idxBl(style_B3HW)
            style_BLCvae = self.quantize_local.msBllist_to_BlCvae(ms_style_idx)
            ms_content_idx = self.vae_local.img_to_idxBl(content_B3HW)
            content_BLCvae = self.quantize_local.msBllist_to_BlCvae(ms_content_idx)

            logits_BLV = self.var_wo_ddp(x_BLCv_wo_first_l, style_BLCvae, content_BLCvae, style_B3HW, content_B3HW)
            L_mean += self.val_loss(logits_BLV.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(
                logits_BLV[:, -self.last_l :].reshape(-1, V), gt_BL[:, -self.last_l :].reshape(-1)
            ) * B
            acc_mean += (logits_BLV.argmax(dim=-1) == gt_BL).sum() * (100 / gt_BL.shape[1])
            acc_tail += (
                logits_BLV[:, -self.last_l :].argmax(dim=-1) == gt_BL[:, -self.last_l :]
            ).sum() * (100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)

        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time() - start

    def train_step(
        self,
        it: int,
        g_it: int,
        stepping: bool,
        metric_lg: MetricLogger,
        tb_lg: TensorboardLogger,
        inp_B3HW: FTen,
        style_B3HW: FTen,
        content_B3HW: FTen,
        prog_si: int,
        prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1:
                self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / max(prog_wp_it, 1), 1), 0.01)
        if self.first_prog:
            prog_wp = 1
        if prog_si == len(self.patch_nums) - 1:
            prog_si = -1

        inp_B3HW = inp_B3HW.float()
        style_B3HW = style_B3HW.float()
        content_B3HW = content_B3HW.float()
        B = inp_B3HW.shape[0]
        V = self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping

        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

        ms_style_idx = self.vae_local.img_to_idxBl(style_B3HW)
        style_BLCvae = self.quantize_local.msBllist_to_BlCvae(ms_style_idx)
        ms_content_idx = self.vae_local.img_to_idxBl(content_B3HW)
        content_BLCvae = self.quantize_local.msBllist_to_BlCvae(ms_content_idx)

        with self.var_opt.amp_ctx:
            logits_BLV = self.var(x_BLCv_wo_first_l, style_BLCvae, content_BLCvae, style_B3HW, content_B3HW)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            if prog_si >= 0:
                bg, ed = self.begin_ends[prog_si]
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:
                lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()

        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)

        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:
                Ltail = acc_tail = -1
            else:
                Ltail = self.val_loss(
                    logits_BLV.data[:, -self.last_l :].reshape(-1, V), gt_BL[:, -self.last_l :].reshape(-1)
                ).item()
                acc_tail = (pred_BL[:, -self.last_l :] == gt_BL[:, -self.last_l :]).float().mean().item() * 100
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=(grad_norm or 0.0))

        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class)
            prob_per_class /= prob_per_class.sum()
            cluster_usage = (prob_per_class > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                kw = {"z_voc_usage": cluster_usage}
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si:
                        break
                    pred = logits_BLV.data[:, bg:ed].reshape(-1, V)
                    tar = gt_BL[:, bg:ed].reshape(-1)
                    kw[f"acc_{self.resos[si]}"] = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    kw[f"L_{self.resos[si]}"] = self.val_loss(pred, tar).item()
                tb_lg.update(head="AR_iter_loss", step=g_it, **kw)
                tb_lg.update(
                    head="AR_iter_schedule",
                    step=g_it,
                    prog_a_reso=self.resos[max(prog_si, 0)],
                    prog_si=prog_si,
                    prog_wp=prog_wp,
                )

        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    def get_config(self):
        return {
            "patch_nums": self.patch_nums,
            "resos": self.resos,
            "label_smooth": self.label_smooth,
            "prog_it": self.prog_it,
            "last_prog_si": self.last_prog_si,
            "first_prog": self.first_prog,
        }

    def state_dict(self):
        state = {"config": self.get_config()}
        for name in ("var_wo_ddp", "vae_local", "var_opt"):
            module = getattr(self, name)
            if module is not None:
                if hasattr(module, "_orig_mod"):
                    module = module._orig_mod
                state[name] = module.state_dict()
        return state

    def load_state_dict(self, state, strict=True, skip_vae=False):
        for name in ("var_wo_ddp", "vae_local", "var_opt"):
            if skip_vae and "vae" in name:
                continue
            module = getattr(self, name)
            if module is None:
                continue
            if hasattr(module, "_orig_mod"):
                module = module._orig_mod
            ret = module.load_state_dict(state[name], strict=strict)
            if ret is not None:
                missing, unexpected = ret
                print(f"[Trainer.load_state_dict] {name} missing: {missing}")
                print(f"[Trainer.load_state_dict] {name} unexpected: {unexpected}")

        config = state.get("config", {})
        self.prog_it = config.get("prog_it", 0)
        self.last_prog_si = config.get("last_prog_si", -1)
        self.first_prog = config.get("first_prog", True)
