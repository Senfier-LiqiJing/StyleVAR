import torch
import torch.nn as nn
from torchvision import models

class AdaINBaseline:
    """
    AdaIN Style Transfer Baseline.
    Paper: Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (Huang et al.)
    """
    def __init__(self, device='cuda', decoder_path="/home/.cache/torch/hub/checkpoints/decoder.pth"):
        self.device = device
        
        # 1. Load VGG Encoder (Pretrained on ImageNet)
        # We use features up to relu4_1
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.encoder = nn.Sequential(*list(vgg.children())[:21])
        self.encoder.to(device).eval()
        
        # 2. Define Decoder (Mirror of Encoder)
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3))
        ).to(device)
        
        # Load decoder weights
        if decoder_path:
            # map_location ensures we load to the correct device directly
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        else:
            print("WARNING: No decoder weights loaded for AdaIN. Output will be noise.")

    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        # N, C, H, W
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adain(self, content_feat, style_feat):
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)
        
        normalized_feat = (content_feat - content_mean) / content_std
        return normalized_feat * style_std + style_mean

    def run(self, content_tensor, style_tensor, alpha=1.0):
        """
        Args:
            content_tensor: torch.Tensor, shape (N, 3, H, W), range [0, 1]
            style_tensor: torch.Tensor, shape (N, 3, H, W), range [0, 1]
            alpha: float (0.0 - 1.0), interpolation strength
        Returns:
            output_tensor: torch.Tensor, shape (N, 3, H, W), range [0, 1]
        """
        # Ensure inputs are on the correct device
        # Assuming input comes from Dataloader as (N, C, H, W)
        c_tensor = content_tensor.to(self.device)
        s_tensor = style_tensor.to(self.device)
        
        with torch.no_grad():
            c_feat = self.encoder(c_tensor)
            s_feat = self.encoder(s_tensor)
            
            # AdaIN Core Operation
            t = self.adain(c_feat, s_feat)
            t = alpha * t + (1 - alpha) * c_feat
            
            # Decode
            g_t = self.decoder(t)
        
        # Post-process: Just clamp to valid image range, keep as Tensor
        g_t = g_t.clamp(0, 1)
        
        return g_t