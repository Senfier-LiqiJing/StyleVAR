import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:  # graceful fallback when huggingface_hub isn't installed
    class PyTorchModelHubMixin:
        pass
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import dist
from models.style_basic_var import AdaLNBeforeHead, AdaLNCrossAttn, make_lora_linear
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
import torchvision.models as models

class SharedAdaLin(nn.Module):
    def __init__(self, in_features, out_features, lora_cfg=None):
        super().__init__()
        self.linear = make_lora_linear(in_features, out_features, bias=True, lora_cfg=lora_cfg)
    
    def forward(self, cond_BD):
        weight = self.linear.get_base_layer().weight if hasattr(self.linear, 'get_base_layer') else self.linear.weight
        C = weight.shape[0] // 6
        return self.linear(cond_BD).view(-1, 1, 6, C)   # B16C


class StyleVAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1, style_enc_dim = 512,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
        alpha_nums = (0.2,0.3,0.4,0.4,0.5,0.5,0.6,0.6,0.7,0.8), # 10 alpha numbers
        lora_rank: int = 0, lora_alpha: float = 1.0, lora_dropout: float = 0.0,
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
        self.lora_cfg = None if lora_rank <= 0 else {
            'r': lora_rank,
            'alpha': lora_alpha,
            'dropout': lora_dropout,
            'adapter_name': 'style_lora',
        }
        self.style_encoder = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).children())[:-1])
        self.content_encoder = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).children())[:-1])
        self.feat_emb = nn.Linear(self.style_enc_dim, self.C, bias=True)
        with torch.no_grad():
            self.feat_emb.weight.zero_()
            if self.feat_emb.bias is not None:
                self.feat_emb.bias.zero_()

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = make_lora_linear(self.Cvae, self.C, bias=True, lora_cfg=self.lora_cfg)
        
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
        self.shared_ada_lin = nn.Sequential(
            nn.SiLU(inplace=False),
            SharedAdaLin(self.D, 6*self.C, lora_cfg=self.lora_cfg),
        ) if shared_aln else nn.Identity()
        
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
                lora_cfg=self.lora_cfg,
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
        self.head = make_lora_linear(self.C, self.V, bias=True, lora_cfg=self.lora_cfg)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int,
        style_img: torch.Tensor, content_img: torch.Tensor, # B,3,H,W in [0,1]
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        
        # fix g_seed for stable generation
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        # style-var relys on input images for generation, not class
        #if label_B is None:
        #    label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        #elif isinstance(label_B, int):
        #    label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        # style-var relys on input images for generation
        # sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        sos = self.feat_emb(self.content_encoder(content_img).squeeze(-1).squeeze(-1))
        cond_BD = self.feat_emb(self.style_encoder(style_img).squeeze(-1).squeeze(-1))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC  # lvl_embed: 1L -> 1LC, pos_1LC:1LC, lvl_pos:1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        # encode the style & content image
        ms_style_idx = self.vae_proxy[0].img_to_idxBl(style_img)
        ms_style_BlCv = self.vae_quant_proxy[0].msBllist_to_BlCv_list(ms_style_idx)
        ms_style_BlC = [self.word_embed(item) for item in ms_style_BlCv]
        ms_content_idx = self.vae_proxy[0].img_to_idxBl(content_img)
        ms_content_BlCv = self.vae_quant_proxy[0].msBllist_to_BlCv_list(ms_content_idx)
        ms_content_BlC = [self.word_embed(item) for item in ms_content_BlCv]
        
        cur_L = 0
        for idx, style_BlC in enumerate(ms_style_BlC):
            style_BlC += lvl_pos[:,cur_L:cur_L+self.patch_nums[idx]**2]
            ms_style_BlC[idx] = style_BlC.expand(2*B,self.patch_nums[idx]**2,-1)
            cur_L += self.patch_nums[idx]**2
        
        cur_L = 0
        for idx, content_BlC in enumerate(ms_content_BlC):
            content_BlC += lvl_pos[:,cur_L:cur_L+self.patch_nums[idx]**2]
            ms_content_BlC[idx] = content_BlC.expand(2*B,self.patch_nums[idx]**2,-1)
            cur_L += self.patch_nums[idx]**2

        cur_L = 0
        # f_hat: B,Cvae,16,16
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1 # 0,1/9,2/9,3/9,...
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNCrossAttn.forward
            for b in self.blocks:
                x = b(x=x, style=ms_style_BlC[si], content=ms_content_BlC[si], cond_BD=cond_BD_or_gss, attn_bias = None, alpha = self.alpha_nums[si])
                # x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD) # C -> D(C) -> V    shape:2B, l, V
            
            # t is defined to control the power of condition in terms of general case
            t = cfg * ratio # cfg = 1.5
            # the previous and latter B batch are conditional and unconditional for generation.
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            # get top-k or top-p logits, which is the index for codebook, and the output size is B,l
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                # for each element in array idx_bl, it is expressed by Cvae dimensions
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
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
        with torch.amp.autocast(device_type='cuda', enabled=False):
            #label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            #sos = cond_BD = self.class_emb(label_B)
            sos = self.feat_emb(self.content_encoder(content_img).squeeze(-1).squeeze(-1))
            cond_BD = self.feat_emb(self.style_encoder(style_img).squeeze(-1).squeeze(-1))
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            style_BLC = self.word_embed(style_BLCvae)
            content_BLC = self.word_embed(content_BLCvae)

            if self.prog_si == 0:
                x_BLC = sos
            else:
                x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
            style_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
            content_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
            x_BLC = x_BLC[:, :ed]
            style_BLC = style_BLC[:, :ed]
            content_BLC = content_BLC[:, :ed]

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

    # ====================== pretrained & freezing helpers ======================
    @staticmethod
    def _extract_var_state_dict(raw_ckpt):
        def looks_like_state(d: dict):
            if not isinstance(d, dict):
                return False
            return any(k.startswith('blocks.') or k.startswith('word_embed') for k in d.keys())

        if looks_like_state(raw_ckpt):
            return raw_ckpt
        if isinstance(raw_ckpt, dict):
            for key in ('var_wo_ddp', 'var', 'model', 'module', 'state_dict'):
                if key in raw_ckpt and looks_like_state(raw_ckpt[key]):
                    return raw_ckpt[key]
            if 'trainer' in raw_ckpt and isinstance(raw_ckpt['trainer'], dict):
                for key in ('var_wo_ddp', 'var', 'model', 'module', 'state_dict'):
                    if key in raw_ckpt['trainer'] and looks_like_state(raw_ckpt['trainer'][key]):
                        return raw_ckpt['trainer'][key]
        raise ValueError('Cannot extract VAR weights from checkpoint; please provide a dict-like state_dict.')

    def _load_from_var_state(self, var_state: dict, zero_unmatched: bool = True):
        target_state = self.state_dict()

        # normalize keys
        normalized = {}
        for k, v in var_state.items():
            nk = k.replace('module.', '')
            normalized[nk] = v

        loaded_keys, zeroed_keys = set(), []

        loaded_pairs = []

        def try_copy(src_key: str, dst_key: str):
            nonlocal loaded_keys
            candidates = [dst_key]
            if dst_key.endswith('.weight'):
                candidates.append(dst_key.replace('.weight', '.base_layer.weight'))
            if dst_key.endswith('.bias'):
                candidates.append(dst_key.replace('.bias', '.base_layer.bias'))
            for cand in candidates:
                if cand not in target_state:
                    continue
                if target_state[cand].shape != normalized[src_key].shape:
                    continue
                target_state[cand] = normalized[src_key]
                loaded_keys.add(cand)
                loaded_pairs.append((src_key, cand, tuple(target_state[cand].shape)))
                return

        for k in normalized.keys():
            if not k.startswith('blocks.') or '.attn.' not in k:
                continue
            if '.attn.mat_qkv.weight' in k:
                prefix = k.replace('.attn.mat_qkv.weight', '.attn.')
                try_copy(k, prefix + 'mat_qkv_guide.weight')
                try_copy(k, prefix + 'mat_qkv_target.weight')
            elif '.attn.q_bias' in k:
                prefix = k.replace('.attn.q_bias', '.attn.')
                try_copy(k, prefix + 'q_bias_guide')
                try_copy(k, prefix + 'q_bias_target')
            elif '.attn.v_bias' in k:
                prefix = k.replace('.attn.v_bias', '.attn.')
                try_copy(k, prefix + 'v_bias_guide')
                try_copy(k, prefix + 'v_bias_target')
            elif '.attn.zero_k_bias' in k:
                prefix = k.replace('.attn.zero_k_bias', '.attn.')
                try_copy(k, prefix + 'zero_k_bias_guide')
                try_copy(k, prefix + 'zero_k_bias_target')

        for k, v in normalized.items():
            if k in loaded_keys:
                continue
            try_copy(k, k)

        self.load_state_dict(target_state, strict=False)

        if zero_unmatched:
            for name, param in self.named_parameters():
                if name in loaded_keys:
                    continue
                if name.startswith('style_encoder') or name.startswith('content_encoder'):
                    continue
                if name.startswith('vae_proxy') or name.startswith('vae_quant_proxy'):
                    continue
                with torch.no_grad():
                    param.zero_()
                    zeroed_keys.append(name)

        self._loaded_param_names = loaded_keys
        self._loaded_param_pairs = loaded_pairs
        self._zero_initialized_param_names = set(zeroed_keys)
        return loaded_keys, zeroed_keys, loaded_pairs

    def apply_training_policy(self, freeze_backbone: bool = True):
        loaded = getattr(self, '_loaded_param_names', set())
        zeroed = set(getattr(self, '_zero_initialized_param_names', set()))

        frozen_names, frozen_loaded_base = [], []
        zero_trainable, lora_trainable, other_trainable = [], [], []

        for name, param in self.named_parameters():
            is_backbone = name.startswith('style_encoder') or name.startswith('content_encoder')
            is_lora = 'lora_' in name
            if freeze_backbone and is_backbone:
                param.requires_grad = False
                frozen_names.append(name)
                continue
            if is_lora:
                param.requires_grad = True
                lora_trainable.append(name)
                continue
            # New (unmatched) non-LoRA params should train: force requires_grad=True if they were zero-initialized
            if name in zeroed:
                param.requires_grad = True
                zero_trainable.append(name)
                continue
            if name in loaded:
                param.requires_grad = False
                frozen_loaded_base.append(name)
                continue
            param.requires_grad = True
            other_trainable.append(name)

        if hasattr(self, 'vae_proxy') and len(self.vae_proxy) > 0:
            for p_name, p in self.vae_proxy[0].named_parameters():
                p.requires_grad = False
                frozen_names.append(f'vae_proxy.{p_name}')

        self._param_policy_info = {
            'frozen': frozen_names,
            'frozen_loaded_base': frozen_loaded_base,
            'zero_trainable': zero_trainable,
            'lora_trainable': lora_trainable,
            'other_trainable': other_trainable,
        }

    def log_param_report(self, preview: int = 12):
        param_dict = dict(self.named_parameters())
        info = getattr(self, '_param_policy_info', {})

        def _section(title, names):
            total_params = sum(param_dict[n].numel() for n in names if n in param_dict)
            shown = names[:preview]
            print(f'[{title}] count={len(names)}, #params={total_params}, showing first {len(shown)}', flush=True)
            for n in shown:
                p = param_dict.get(n, None)
                shape = tuple(p.shape) if p is not None else 'N/A'
                grad = getattr(p, 'requires_grad', None)
                print(f'  - {n:60s} | shape={shape} | requires_grad={grad}', flush=True)
            if len(names) > preview:
                print(f'  ... ({len(names) - preview} more)', flush=True)

        total = sum(p.numel() for p in param_dict.values())
        trainable = sum(p.numel() for p in param_dict.values() if p.requires_grad)
        print(f'[param report] total_params={total}, trainable_params={trainable}', flush=True)
        _section('frozen', info.get('frozen', []))
        _section('frozen_loaded_base', info.get('frozen_loaded_base', []))
        _section('zero_trainable', info.get('zero_trainable', []))
        _section('lora_trainable', info.get('lora_trainable', []))
        _section('other_trainable', info.get('other_trainable', []))
        if hasattr(self, 'vae_proxy') and len(self.vae_proxy) > 0:
            vae_params = list(self.vae_proxy[0].named_parameters())
            vae_count = sum(p.numel() for _, p in vae_params)
            shown = vae_params[:preview]
            print(f'[vae_proxy frozen] count={len(vae_params)}, #params={vae_count}, showing first {len(shown)}', flush=True)
            for n, p in shown:
                print(f'  - vae_proxy.{n:56s} | shape={tuple(p.shape)} | requires_grad={p.requires_grad}', flush=True)
            if len(vae_params) > preview:
                print(f'  ... ({len(vae_params) - preview} more)', flush=True)

    @classmethod
    def from_pretrained(
        cls,
        vae_local: VQVAE,
        pretrained_path: str,
        map_location='cpu',
        zero_unmatched: bool = True,
        freeze_resnet: bool = True,
        freeze_backbone: bool = True,
        **kwargs,
    ):
        raw_ckpt = torch.load(pretrained_path, map_location=map_location)
        var_state = cls._extract_var_state_dict(raw_ckpt)
        model = cls(vae_local=vae_local, **kwargs)
        loaded, zeroed, pairs = model._load_from_var_state(var_state, zero_unmatched=zero_unmatched)
        log_lines = ['[load mapping from VAR]']
        for src, dst, shape in pairs:
            log_lines.append(f'  {src:60s} -> {dst:60s} | shape={shape}')
        log_path = os.path.join(os.getcwd(), 'load_mapping.log')
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(log_lines))
            print(f'[load mapping] entries={len(pairs)}, written to {log_path}', flush=True)
        except Exception as e:
            print(f'[load mapping] failed to write log ({e}); fallback to stdout.', flush=True)
            for line in log_lines:
                print(line, flush=True)
        model.apply_training_policy(freeze_backbone=True)
        model.log_param_report()
        return model, {'loaded': loaded, 'zeroed': zeroed, 'pairs': pairs, 'policy': getattr(model, '_param_policy_info', {})}


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
        alpha_nums = (0.2,0.3,0.4,0.4,0.5,0.5,0.6,0.6,0.7,0.8) # 10 alpha numbers
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
