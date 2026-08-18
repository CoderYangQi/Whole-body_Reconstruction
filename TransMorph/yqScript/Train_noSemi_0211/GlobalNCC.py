import torch

class GlobalNCC(torch.nn.Module):
    """
    Global Normalized Cross-Correlation (NCC) using PyTorch.
    """

    def forward(self, y_true, y_pred):
        """
        计算全局图像之间的 NCC。
        参数:
            y_true: torch.Tensor - 真实值 (shape: [batch_size, *vol_shape])
            y_pred: torch.Tensor - 预测值 (shape: [batch_size, *vol_shape])
        返回:
            ncc_value: torch.Tensor - 全局 NCC 值。
        """
        # 确保输入的形状一致
        assert y_true.shape == y_pred.shape, "Input shapes must match."

        # 计算均值
        mean_y_true = torch.mean(y_true.float())
        mean_y_pred = torch.mean(y_pred.float())

        # 计算分子和分母
        numerator = torch.sum((y_true - mean_y_true) * (y_pred - mean_y_pred))
        denominator = torch.sqrt(
            torch.sum((y_true - mean_y_true) ** 2) * torch.sum((y_pred - mean_y_pred) ** 2)
        )

        # 计算 NCC
        if denominator == 0:
            return torch.tensor(0.0, device=y_true.device)
        ncc_value = numerator / denominator
        return ncc_value