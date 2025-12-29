from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch

def calculate_ssim(pred_img, target_img, device='cuda'):
    """
    Calculates SSIM between two images.
    Args:
        pred_img: Tensor (B, 3, H, W), range [0, 1]
        target_img: Tensor (B, 3, H, W), range [0, 1]
    Returns:
        SSIM score (Higher is better, max 1.0)
    """
    # Initialize metric
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    with torch.no_grad():
        score = ssim(pred_img, target_img)
        
    return score.item()

# Usage Example
# ssim_score = calculate_ssim(generated_img, content_img, device='cuda')
# print(f"SSIM Score: {ssim_score}")