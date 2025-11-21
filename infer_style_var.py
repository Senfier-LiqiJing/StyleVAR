import argparse
import os
from typing import Tuple

import torch
from torchvision import transforms
from torchvision.utils import save_image

from models.vqvae import VQVAE
from models.style_var import StyleVAR
from utils.data import normalize_01_into_pm1


def build_transforms(reso: int, mid_reso_mult: float = 1.125):
    mid_reso = round(reso * mid_reso_mult)
    return transforms.Compose([
        transforms.Resize(mid_reso, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(reso),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])


def load_models(
    ckpt_path: str,
    vae_ckpt_path: str,
    device: torch.device,
    depth: int = 20,
    patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
):
    """Load VAE and StyleVAR from local checkpoints."""
    print(f'[info] loading trainer checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)
    trainer_state = ckpt.get('trainer', {})
    args_state = ckpt.get('args', {})

    # allow args override from ckpt
    depth = args_state.get('depth', depth)
    patch_nums = tuple(args_state.get('patch_nums', patch_nums))

    # instantiate VAE & load weights
    print(f'[info] instantiating VQVAE depth={len(patch_nums)}, z=32, vocab=4096')
    vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums).to(device)
    if vae_ckpt_path:
        print(f'[info] loading VAE weights from {vae_ckpt_path}')
        vae_sd = torch.load(vae_ckpt_path, map_location=device)
        vae.load_state_dict(vae_sd, strict=True)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # instantiate StyleVAR
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24
    print(f'[info] instantiating StyleVAR depth={depth}, embed_dim={width}, num_heads={heads}')
    model = StyleVAR(
        vae_local=vae,
        num_classes=args_state.get('num_classes', 1000),
        depth=depth,
        embed_dim=width,
        num_heads=heads,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=dpr,
        norm_eps=1e-6,
        shared_aln=args_state.get('saln', False),
        cond_drop_rate=0.1,
        style_enc_dim=args_state.get('style_enc_dim', 512),
        attn_l2_norm=args_state.get('anorm', True),
        patch_nums=patch_nums,
        flash_if_available=args_state.get('fuse', True),
        fused_if_available=args_state.get('fuse', True),
        alpha_nums=args_state.get('alpha_nums', (0.2, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8)),
        lora_rank=args_state.get('lora_r', 0),
        lora_alpha=args_state.get('lora_alpha', 1.0),
        lora_dropout=args_state.get('lora_dropout', 0.0),
    ).to(device)

    # load weights from trainer state if available
    var_state = trainer_state.get('var_wo_ddp') or trainer_state.get('var') or ckpt.get('model') or ckpt
    if isinstance(var_state, dict):
        print(f'[info] loading StyleVAR weights from trainer state')
        missing, unexpected = model.load_state_dict(var_state, strict=False)
        if missing or unexpected:
            print(f'[warn] load_state_dict: missing={missing}, unexpected={unexpected}')
    model.eval()
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad_(False)
        else:
            p.requires_grad_(False)

    return vae, model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='StyleVAR inference')
    parser.add_argument('--ckpt', required=True, help='Path to StyleVAR checkpoint (trainer ckpt).')
    parser.add_argument('--vae_ckpt', default='', help='Path to VQVAE checkpoint.')
    parser.add_argument('--content', required=True, help='Path to content image.')
    parser.add_argument('--style', required=True, help='Path to style image.')
    parser.add_argument('--out', default='out.png', help='Output image path.')
    parser.add_argument('--reso', type=int, default=256, help='Final resolution (square).')
    parser.add_argument('--cfg', type=float, default=1.5, help='Classifier-free guidance scale.')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling.')
    parser.add_argument('--top_p', type=float, default=0.0, help='Top-p sampling.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    tfm = build_transforms(args.reso)
    from PIL import Image
    try:
        content_img = tfm(Image.open(args.content).convert('RGB')).unsqueeze(0).to(device)
        style_img = tfm(Image.open(args.style).convert('RGB')).unsqueeze(0).to(device)
    except Exception as e:
        raise RuntimeError(f'Failed to load input images: {e}')

    # build & load
    try:
        _, model = load_models(
            ckpt_path=args.ckpt,
            vae_ckpt_path=args.vae_ckpt,
            device=device,
        )
    except Exception as e:
        raise RuntimeError(f'Failed to build/load models: {e}')

    # inference
    try:
        print('[info] running autoregressive inference...')
        out = model.autoregressive_infer_cfg(
            B=1,
            style_img=style_img,
            content_img=content_img,
            g_seed=args.seed,
            cfg=args.cfg,
            top_k=args.top_k,
            top_p=args.top_p,
            more_smooth=False,
        )
    except Exception as e:
        raise RuntimeError(f'Inference failed: {e}')

    try:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        save_image(out, args.out)
        print(f'[done] saved to {args.out}')
    except Exception as e:
        raise RuntimeError(f'Failed to save output image: {e}')

if __name__ == '__main__':
    main()