import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class VGGStyleContentLoss(nn.Module):
    """
    Calculates Content Loss and Style Loss using a pretrained VGG19 network.
    Reference: Gatys et al., "A Neural Algorithm of Artistic Style".
    """
    def __init__(self, device='cuda'):
        super().__init__()
        # Load VGG19 pretrained on ImageNet, set to eval mode
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.to(device).eval()
        
        # Freeze parameters since we only use it as a feature extractor
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.device = device
        
        # Standard layers used for Style Transfer (Gatys et al.)
        # '0': conv1_1, '5': conv2_1, '10': conv3_1, '19': conv4_1, '21': conv4_2, '28': conv5_1
        self.content_layers = {'21': 'content'} # usually conv4_2
        self.style_layers = {'0': 'style_1', '5': 'style_2', '10': 'style_3', '19': 'style_4', '28': 'style_5'}
        
        # Normalization for VGG (ImageNet stats)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def _normalize(self, tensor):
        """ Normalize tensor (0-1) with ImageNet mean/std """
        return (tensor - self.mean) / self.std

    def _gram_matrix(self, input_tensor):
        """ Calculates Gram Matrix: (b, c, h, w) -> (b, c, c) """
        b, c, h, w = input_tensor.size()
        features = input_tensor.view(b, c, h * w)
        # BMM: Batch Matrix Multiplication
        G = torch.bmm(features, features.transpose(1, 2))
        # Normalize by total number of elements
        return G.div(c * h * w)

    def forward(self, generated_img, content_target, style_target):
        """
        Args:
            generated_img: Tensor (B, 3, H, W), range [0, 1]
            content_target: Tensor (B, 3, H, W), range [0, 1]
            style_target: Tensor (B, 3, H, W), range [0, 1]
        Returns:
            content_loss, style_loss (scalar values)
        """
        # 1. Normalize inputs
        gen = self._normalize(generated_img)
        con = self._normalize(content_target)
        sty = self._normalize(style_target)
        
        content_loss = 0.0
        style_loss = 0.0
        
        # 2. Extract features layer by layer
        # We need to pass images through VGG layers manually to extract intermediate outputs
        # To avoid multiple passes, we can hook or iterate. Here we iterate for clarity.
        
        # Prepare dictionaries to store features
        gen_features = {}
        con_features = {}
        sty_features = {}
        
        x_gen, x_con, x_sty = gen, con, sty
        
        # Iterate through the model layers
        for name, layer in self.vgg.named_children():
            x_gen = layer(x_gen)
            x_con = layer(x_con)
            x_sty = layer(x_sty)
            
            if name in self.content_layers:
                content_loss += F.mse_loss(x_gen, x_con)
            
            if name in self.style_layers:
                gram_gen = self._gram_matrix(x_gen)
                gram_sty = self._gram_matrix(x_sty)
                style_loss += F.mse_loss(gram_gen, gram_sty)
                
        return content_loss, style_loss

# Usage Example
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# calculator = VGGStyleContentLoss(device)
# c_loss, s_loss = calculator(gen_tensor, content_tensor, style_tensor)
# print(f"Content Loss: {c_loss.item()}, Style Loss: {s_loss.item()}")