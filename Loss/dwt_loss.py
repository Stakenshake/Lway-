import torch
import torch.nn.functional as F
import pywt
from lpips import LPIPS

class WeightedHighFrequencyLoss(torch.nn.Module):
    def __init__(self, wavelet='haar'):
        super(WeightedHighFrequencyLoss, self).__init__()
        self.wavelet = wavelet
        self.lpips_loss = LPIPS(net='alex').cuda()  # 初始化 LPIPS 损失

    def forward(self, I_pred, I_gt):
        assert I_pred.shape == I_gt.shape, "Predicted and ground truth images must have the same shape."
        batch_size, channels, height, width = I_pred.size()

        # 初始化高频权重图
        weight_maps = []
        for b in range(batch_size):
            for c in range(channels):
                # 转为 NumPy 格式（小波变换使用 NumPy）
                pred_np = I_pred[b, c].detach().cpu().numpy()
                gt_np = I_gt[b, c].detach().cpu().numpy()

                # 小波分解提取高频分量
                _, (cH_pred, cV_pred, cD_pred) = pywt.dwt2(pred_np, wavelet=self.wavelet)
                _, (cH_gt, cV_gt, cD_gt) = pywt.dwt2(gt_np, wavelet=self.wavelet)

                # 高频分量的 L2 范数
                high_freq_pred = (cH_pred ** 2 + cV_pred ** 2 + cD_pred ** 2) ** 0.5
                high_freq_gt = (cH_gt ** 2 + cV_gt ** 2 + cD_gt ** 2) ** 0.5

                # 差值生成高频权重图
                weight_map = torch.tensor(abs(high_freq_pred - high_freq_gt), device=I_pred.device, dtype=torch.float32)
                weight_maps.append(weight_map)

        # 合并权重图并归一化到 [0, 1]
        weight_map = torch.stack(weight_maps, dim=0)  # [B*C, H/2, W/2]
        weight_map = F.interpolate(weight_map.unsqueeze(1), size=(height, width), mode='bilinear', align_corners=False)
        weight_map = (weight_map - weight_map.min()) / (weight_map.max() - weight_map.min() + 1e-8)

