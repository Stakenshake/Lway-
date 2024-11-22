import torch
import torch.nn as nn
import lpips
class RecLoss(nn.Module):
    def __init__(self, weight_l1=1.0, weight_lpips=1.0, device='cuda'):
        super(RecLoss, self).__init__()
        # LPIPS loss
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)  # 使用VGG网络计算LPIPS
        self.weight_l1 = weight_l1
        self.weight_lpips = weight_lpips

    def forward(self, IHR, I_hat_LR):
        """
        Compute the L1 and LPIPS loss combined.
        Args:
            IHR (Tensor): Ground truth high-resolution image (batch, C, H, W)
            I_hat_LR (Tensor): Predicted low-resolution image (batch, C, H, W)
        Returns:
            Tensor: Total loss
        """
        # L1 Loss (Mean Absolute Error)
        l1_loss = torch.abs(IHR - I_hat_LR).mean()

        # LPIPS Loss
        lpips_loss = self.lpips_loss(IHR, I_hat_LR)

        # Total loss
        total_loss = self.weight_l1 * l1_loss + self.weight_lpips * lpips_loss

        return total_loss
