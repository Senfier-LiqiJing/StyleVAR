import os
import sys
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from content_style_loss import VGGStyleContentLoss
from LPIPS import LPIPSCalculator
from SSIM import calculate_ssim
from adain import AdaINBaseline

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.vqvae import VQVAE, VectorQuantizer2
vae_local = VQVAE(ch=160).cuda()
checkpoint_path = r"/home/PML-Project/checkpoints/vae_ch160v4096z32.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)
vae_local.load_state_dict(checkpoint)

from models.style_var import StyleVAR
var_model = StyleVAR(vae_local=vae_local,depth=20,embed_dim=1280,attn_l2_norm=True,num_heads=20).cuda()
checkpoint_path = r"/home/PML-Project/local_output/style_var_d20_11_20_12_33.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)
var_model.load_state_dict(checkpoint)

class EvaluationDataset(Dataset):
    def __init__(self, root_dir, sample_files, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = sample_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pair_filename = self.files[idx] 
        content_name, style_name_raw = pair_filename.split('&&')
        
        if style_name_raw.endswith('.png'):
            if os.path.exists(os.path.join(self.root_dir, 'style', style_name_raw[:-4])):
                style_name = style_name_raw[:-4]
            else:
                style_name = style_name_raw
        else:
            style_name = style_name_raw

        content_path = os.path.join(self.root_dir, 'content', content_name)
        style_path = os.path.join(self.root_dir, 'style', style_name)

        c_img = Image.open(content_path).convert('RGB')
        s_img = Image.open(style_path).convert('RGB')

        if self.transform:
            c_img = self.transform(c_img)
            s_img = self.transform(s_img)

        return c_img, s_img

def get_dataloader(data_path, num_samples=500, batch_size=8):
    target_dir = os.path.join(data_path, 'target')
    all_files = [f for f in os.listdir(target_dir) if '&&' in f]
    
    if len(all_files) < num_samples:
        sampled_files = all_files
    else:
        sampled_files = random.sample(all_files, num_samples)
        
    print(f"Selected {len(sampled_files)} pairs for evaluation.")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = EvaluationDataset(data_path, sampled_files, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader

def evaluate_model(generator, dataloader, device):
    vgg_calc = VGGStyleContentLoss().to(device)
    lpips_calc = LPIPSCalculator(device=device)
    
    # Trackers
    metrics = {
        'style_loss': 0.0,
        'content_loss': 0.0,
        'lpips': 0.0,
        'ssim': 0.0,
        'inference_time': 0.0
    }
    total_images = 0
    
    print("Starting Inference...")
    
    with torch.no_grad():
        for i, (content, style) in enumerate(dataloader):
            content = content.to(device)
            style = style.to(device)
            B = content.shape[0]
            
            # 1. Inference
            torch.cuda.synchronize()
            t_start = time.time()
            
            if hasattr(generator,"autoregressive_infer_cfg"):
                generated = generator.autoregressive_infer_cfg(
                    B=B, style_img=style, content_img=content, 
                    top_k=900, top_p=0.96
                )
            else:
                generated = generator.run(content_tensor= content, style_tensor= style)
            
            torch.cuda.synchronize()
            metrics['inference_time'] += (time.time() - t_start)

            # 2. Denormalize to [0, 1] for metrics
            gen_norm = generated.add(1).mul(0.5).clamp(0, 1)
            con_norm = content.add(1).mul(0.5).clamp(0, 1)
            sty_norm = style.add(1).mul(0.5).clamp(0, 1)

            # 3. Calculate Metrics
            # Style Loss: Generated vs Style
            # Content Loss: Generated vs Content
            c_loss, s_loss = vgg_calc(gen_norm, con_norm, sty_norm)
            
            # LPIPS: Generated vs Content
            lpips_val = lpips_calc.calculate(gen_norm, con_norm)
            
            # SSIM: Generated vs Content
            ssim_val = calculate_ssim(gen_norm, con_norm)

            # Accumulate
            metrics['style_loss'] += s_loss.item() * B
            metrics['content_loss'] += c_loss.item() * B
            
            if isinstance(lpips_val, torch.Tensor): lpips_val = lpips_val.item()
            metrics['lpips'] += lpips_val * B
            
            if isinstance(ssim_val, torch.Tensor): ssim_val = ssim_val.item()
            metrics['ssim'] += ssim_val * B
            
            total_images += B
            if (i+1) % 10 == 0: print(f"Evaluated {total_images} samples...")

    # Final Report
    print("\n" + "="*30)
    print(f"Evaluation Report (N={total_images})")
    print("-" * 30)
    print(f"Speed: {total_images / metrics['inference_time']:.2f} FPS")
    print(f"Avg Inference Time: {metrics['inference_time'] / total_images:.4f} s")
    print("-" * 30)
    print(f"Style Preservation (Style Loss) ↓:   {metrics['style_loss'] / total_images:.4f}")
    print(f"Content Preservation (Cont Loss) ↓:  {metrics['content_loss'] / total_images:.4f}")
    print(f"Structure Preservation (SSIM) ↑:     {metrics['ssim'] / total_images:.4f}")
    print(f"Perceptual Distance (LPIPS) ↓:       {metrics['lpips'] / total_images:.4f}")
    print("="*30)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "/home/OmniStyle-150K"
    loader = get_dataloader(data_path, num_samples=500, batch_size=1)
    evaluate_model(var_model, loader, device)
    evaluate_model(AdaINBaseline(),loader,device)