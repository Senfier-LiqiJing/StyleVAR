import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import dist
from models.style_basic_var import AdaLNBeforeHead, AdaLNCrossAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
import torchvision.models as models

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class StyleVAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1, style_enc_dim = 512,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
        alpha_nums = (0.3,0.4,0.5,0.5,0.5,0.5,0.5,0.4,0.3,0.2) # 10 alpha numbers: mid-high, ends-low (preserve detail at fine scales)
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training

        self.patch_nums: Tuple[int] = patch_nums
        self.alpha_nums: Tuple[float] = alpha_nums
        self.alpha_jitter: float = 0.0   # set > 0 to enable alpha jitter during training
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # define style/ content encoder
        self.style_enc_dim = style_enc_dim
        self.style_encoder = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).children())[:-1])
        self.content_encoder = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).children())[:-1])
        self.feat_emb = nn.Linear(self.style_enc_dim,self.C)

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        #self.num_classes = num_classes
        #self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        #self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        # nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNCrossAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer(
        self, B: int,
        style_img: torch.Tensor, content_img: torch.Tensor,   # (B,3,H,W) in [-1,1]
        g_seed: Optional[int] = None, top_k=0, top_p=0.0,
        more_smooth=False,
        style_strength: float = 0.0,
    ) -> torch.Tensor:
        """
        Autoregressive inference **without** classifier-free guidance.
        Works with arbitrary batch size B.

        :param B:  batch size (must match style_img.shape[0] and content_img.shape[0])
        :param style_img:   (B,3,H,W) style reference images in [-1,1]
        :param content_img: (B,3,H,W) content images in [-1,1]
        :param g_seed:  random seed (None = non-deterministic)
        :param top_k:   top-k sampling (0 = disabled)
        :param top_p:   top-p sampling (0.0 = disabled)
        :param more_smooth: use gumbel-softmax instead of argmax (visualization only)
        :param style_strength: in [-1, 1].  0 = default balance;
               +1 = max style (all alphas ↑);  -1 = max content (all alphas ↓).
               Requires the model to be trained with alpha_jitter > 0 for best results.
        :return: generated images (B,3,H,W) in [0,1]
        """
        # Compute effective alphas with user-controlled style-content balance
        if style_strength != 0.0:
            shift = style_strength * 0.5   # maps [-1,1] → [-0.5, +0.5] shift
            effective_alphas = tuple(
                max(0.01, min(0.99, a + shift)) for a in self.alpha_nums)
        else:
            effective_alphas = self.alpha_nums

        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed); rng = self.rng

        # ---- condition embeddings (B, C) ----
        sos     = self.feat_emb(self.content_encoder(content_img).squeeze(-1).squeeze(-1))
        cond_BD = self.feat_emb(self.style_encoder(style_img).squeeze(-1).squeeze(-1))

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC   # (1, L, C)

        # SOS → first token map  (B, first_l, C)
        next_token_map = (
            sos.unsqueeze(1).expand(B, self.first_l, -1)
            + self.pos_start.expand(B, self.first_l, -1)
            + lvl_pos[:, :self.first_l]
        )

        # ---- VQ-VAE multi-scale tokenisation of style & content ----
        ms_style_idx   = self.vae_proxy[0].img_to_idxBl(style_img)
        ms_style_BlCv  = self.vae_quant_proxy[0].msBllist_to_BlCv_list(ms_style_idx)
        ms_style_BlC   = [self.word_embed(item) for item in ms_style_BlCv]

        ms_content_idx  = self.vae_proxy[0].img_to_idxBl(content_img)
        ms_content_BlCv = self.vae_quant_proxy[0].msBllist_to_BlCv_list(ms_content_idx)
        ms_content_BlC  = [self.word_embed(item) for item in ms_content_BlCv]

        # add level + position embeddings  (broadcasts from (1,pn²,C) to (B,pn²,C))
        cur_L = 0
        for idx_s in range(len(self.patch_nums)):
            pn = self.patch_nums[idx_s]
            pos = lvl_pos[:, cur_L:cur_L + pn * pn]            # (1, pn², C)
            ms_style_BlC[idx_s]   = ms_style_BlC[idx_s]   + pos
            ms_content_BlC[idx_s] = ms_content_BlC[idx_s] + pos
            cur_L += pn * pn

        # cumulative feature map  (B, Cvae, max_pn, max_pn)
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        # ---- scale-by-scale autoregressive generation ----
        for blk in self.blocks:
            blk.attn.kv_caching(True)

        cur_L = 0
        for si, pn in enumerate(self.patch_nums):
            cur_L += pn * pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)

            x = next_token_map
            for blk in self.blocks:
                x = blk(x=x, style=ms_style_BlC[si], content=ms_content_BlC[si],
                        cond_BD=cond_BD_or_gss, attn_bias=None,
                        alpha=effective_alphas[si])

            logits_BlV = self.get_logits(x, cond_BD)           # (B, pn², V)

            # ---- sample (no CFG) ----
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1,
            )[:, :, 0]                                          # (B, pn²)

            if not more_smooth:
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)          # (B, pn², Cvae)
            else:
                ratio = si / self.num_stages_minus_1
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False,
                    dim=-1, rng=rng,
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(self.patch_nums), f_hat, h_BChw)

            if si != self.num_stages_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    self.word_embed(next_token_map)
                    + lvl_pos[:, cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                )
                # NO repeat(2,1,1) — no CFG batch doubling

        for blk in self.blocks:
            blk.attn.kv_caching(False)

        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # [-1,1] → [0,1]

    # kept for backward compatibility; delegates to autoregressive_infer (ignoring cfg)
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int,
        style_img: torch.Tensor, content_img: torch.Tensor,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:
        """Legacy wrapper — CFG is not supported (no unconditional training).
        Calls :meth:`autoregressive_infer` directly, ignoring ``cfg``."""
        return self.autoregressive_infer(
            B, style_img, content_img,
            g_seed=g_seed, top_k=top_k, top_p=top_p, more_smooth=more_smooth,
        )
    
    def forward(self, x_BLCv_wo_first_l: torch.Tensor, style_BLCvae: torch.Tensor, content_BLCvae: torch.tensor, style_img:torch.tensor, content_img:torch.tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :param style_BLCvae: multi-scale style feature concatenated (B, self.L, self.Cvae)
        :param content_BLCvae: multi-scale style feature concatenated (B, self.L, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            #label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            #sos = cond_BD = self.class_emb(label_B)
            sos = self.feat_emb(self.content_encoder(content_img).squeeze(-1).squeeze(-1))
            cond_BD = self.feat_emb(self.style_encoder(style_img).squeeze(-1).squeeze(-1))
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: 
                x_BLC = sos
            else: 
                x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
                style_BLC = self.word_embed(style_BLCvae)
                content_BLC = self.word_embed(content_BLCvae)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
            style_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
            content_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        # alpha in training should be a tensor, since multi-stage logits is output simultanously and no single alpha should be allowd.
        # alpha in inference only need to take one scalar.
        alpha_map_tensor = torch.tensor(self.alpha_nums, device=x_BLC.device, dtype=x_BLC.dtype)

        # Alpha jitter: random global shift during training for robustness
        # (also enables user-controllable style_strength at inference)
        if self.training and self.alpha_jitter > 0:
            shift = torch.empty(1, device=x_BLC.device, dtype=x_BLC.dtype).uniform_(
                -self.alpha_jitter, self.alpha_jitter)
            alpha_map_tensor = (alpha_map_tensor + shift).clamp_(0.01, 0.99)

        lvls_1_ed = self.lvl_1L[:, :ed]
        lvls_B_ed = lvls_1_ed.expand(B, -1)
        alpha_tensor_B_ed = alpha_map_tensor[lvls_B_ed]
        alpha_tensor_BLC = alpha_tensor_B_ed.unsqueeze(-1)

        AdaLNCrossAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, style=style_BLC, content=content_BLC , cond_BD=cond_BD_or_gss, attn_bias=attn_bias,alpha = alpha_tensor_BLC)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(StyleVAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,style_enc_dim = 512,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
        alpha_nums = (0.3,0.4,0.5,0.5,0.5,0.5,0.5,0.4,0.3,0.2) # 10 alpha numbers
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,style_enc_dim = style_enc_dim,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            alpha_nums = alpha_nums
        )
