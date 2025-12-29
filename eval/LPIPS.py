import lpips
import torch

class LPIPSCalculator:
    """
    Wrapper for LPIPS metric.
    Paper: The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
    """
    def __init__(self, device='cuda', net='vgg'):
        # net can be 'alex', 'vgg', or 'squeeze'
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.device = device

    def calculate(self, img1, img2):
        """
        Args:
            img1, img2: Tensor (B, 3, H, W), range [0, 1]
        Returns:
            LPIPS distance (Lower is better, meaning more perceptually similar)
        """
        # LPIPS expects input in range [-1, 1]
        img1_norm = img1 * 2 - 1
        img2_norm = img2 * 2 - 1
        
        with torch.no_grad():
            dist = self.loss_fn(img1_norm, img2_norm)
        
        return dist.mean().item()

# Usage Example
# lpips_calc = LPIPSCalculator(device='cuda')
# score = lpips_calc.calculate(generated_img, content_img) # Compare Generation vs Content Source
# print(f"LPIPS Score: {score}")