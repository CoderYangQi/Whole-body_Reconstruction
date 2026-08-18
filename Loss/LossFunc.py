import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn
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
        mean_y_true = torch.mean(y_true)
        mean_y_pred = torch.mean(y_pred)

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

import torch
import torch.nn.functional as F
from torch import nn


class SSIMLoss(nn.Module):
    """
    PyTorch implementation of the SSIM loss function.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, spatial_dims: int = 2):
        """
        Args:
            win_size: size of the Gaussian weighting window.
            k1: stability constant used in the luminance denominator.
            k2: stability constant used in the contrast denominator.
            spatial_dims: if 2, the input shape should be (B,C,H,W); if 3, (B,C,H,W,D).
        """
        super().__init__()
        self.win_size = win_size
        self.k1 = k1
        self.k2 = k2
        self.spatial_dims = spatial_dims

        # Create the Gaussian window
        self.register_buffer(
            "w", torch.ones([1, 1] + [win_size for _ in range(spatial_dims)]) / win_size**spatial_dims
        )
        self.cov_norm = (win_size**2) / (win_size**2 - 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the first image tensor, shape (B,C,H,W) or (B,C,H,W,D).
            y: the second image tensor, shape (B,C,H,W) or (B,C,H,W,D).
            data_range: dynamic range of the input data.

        Returns:
            SSIM loss (1 - SSIM value).
        """
        if x.shape[1] > 1:  # Handle multi-channel input
            if x.shape[1] != y.shape[1]:
                raise ValueError(f"Number of channels in x ({x.shape[1]}) and y ({y.shape[1]}) must be the same.")
            losses = torch.stack(
                [
                    SSIMLoss(self.win_size, self.k1, self.k2, self.spatial_dims)(
                        x[:, i, ...].unsqueeze(1), y[:, i, ...].unsqueeze(1), data_range
                    )
                    for i in range(x.shape[1])
                ]
            )
            return losses.mean()

        # Adjust the data range for convolution
        data_range = data_range[(None,) * (self.spatial_dims + 2)]
        conv = getattr(F, f"conv{self.spatial_dims}d")
        w = self.w.to(x.dtype).to(x.device)

        # # Compute means
        # ux = conv(x, w, padding=self.win_size // 2, groups=1)
        # uy = conv(y, w, padding=self.win_size // 2, groups=1)
        #
        # # Compute variances and covariances
        # uxx = conv(x * x, w, padding=self.win_size // 2, groups=1)
        # uyy = conv(y * y, w, padding=self.win_size // 2, groups=1)
        # uxy = conv(x * y, w, padding=self.win_size // 2, groups=1)
        #
        # vx = self.cov_norm * (uxx - ux**2)
        # vy = self.cov_norm * (uyy - uy**2)
        # vxy = self.cov_norm * (uxy - ux * uy)
        #
        # # Stability constants
        # c1 = (self.k1 * data_range) ** 2
        # c2 = (self.k2 * data_range) ** 2

        c1 = (self.k1 * data_range) ** 2  # stability constant for luminance
        c2 = (self.k2 * data_range) ** 2  # stability constant for contrast
        ux = conv(x, w)  # mu_x
        uy = conv(y, w)  # mu_y
        uxx = conv(x * x, w)  # mu_x^2
        uyy = conv(y * y, w)  # mu_y^2
        uxy = conv(x * y, w)  # mu_xy
        vx = self.cov_norm * (uxx - ux * ux)  # sigma_x
        vy = self.cov_norm * (uyy - uy * uy)  # sigma_y
        vxy = self.cov_norm * (uxy - ux * uy)  # sigma_xy

        # SSIM formula
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denom = (ux**2 + uy**2 + c1) * (vx + vy + c2)
        ssim_map = numerator / denom

        # Compute the loss
        loss = 1 - ssim_map.mean()
        return loss

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class DiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true, num_clus):
        # 转换预测值为概率（适用于二分类或多分类任务）
        y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3, 4])
        union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
        dsc = (2. * intersection) / (union + self.eps)

        # 返回 Dice 损失
        dice_loss = 1 - torch.mean(dsc)
        return dice_loss

class GlobalMutualInformationLoss(_Loss):
    """
    Differentiable global mutual information loss via Parzen windowing method.
    """

    def __init__(
        self,
        kernel_type: str = "gaussian",
        num_bins: int = 23,
        sigma_ratio: float = 0.5,
        reduction: str = "mean",
        smooth_nr: float = 1e-7,
        smooth_dr: float = 1e-7,
    ) -> None:
        """
        Args:
            kernel_type: Type of kernel for Parzen windowing ("gaussian" or "b-spline").
            num_bins: Number of intensity bins.
            sigma_ratio: Sigma ratio for Gaussian kernel.
            reduction: Reduction method ("mean", "sum", or "none").
            smooth_nr: Small constant added to numerator for numerical stability.
            smooth_dr: Small constant added to denominator for numerical stability.
        """
        super().__init__(reduction=reduction)
        if num_bins <= 0:
            raise ValueError("num_bins must be greater than 0")
        self.kernel_type = kernel_type
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio
        self.reduction = reduction
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr

        # Compute bin centers for intensity bins
        self.bin_centers = torch.linspace(0.0, 1.0, num_bins).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_bins)
        sigma = torch.mean(self.bin_centers[0, 0, 1:] - self.bin_centers[0, 0, :-1]) * sigma_ratio
        self.preterm = 1 / (2 * sigma**2)

    def parzen_windowing_gaussian(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parzen windowing with a Gaussian kernel.
        Args:
            img: Input image tensor, shape (B, NDHW).
        Returns:
            Probability distribution for image intensities, shape (B, num_bins).
        """
        img = torch.clamp(img, 0, 1)  # Clamp to [0, 1]
        img = img.reshape(img.shape[0], -1, 1)  # Reshape to (B, num_samples, 1)
        weight = torch.exp(-self.preterm.to(img.device) * (img - self.bin_centers.to(img.device))**2)
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # Normalize
        probability = torch.mean(weight, dim=-2, keepdim=True)  # Compute probability
        return weight, probability

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute global mutual information between `pred` and `target`.
        Args:
            pred: Predicted image, shape (B, NDHW).
            target: Target image, shape (B, NDHW).
        Returns:
            Mutual information loss.
        """
        if target.shape != pred.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        wa, pa = self.parzen_windowing_gaussian(pred)
        wb, pb = self.parzen_windowing_gaussian(target)

        # Joint probability
        pab = torch.bmm(wa.permute(0, 2, 1), wb) / wa.shape[1]  # (B, num_bins, num_bins)
        papb = torch.bmm(pa.permute(0, 2, 1), pb)  # Independent probability

        # Mutual information
        mi = torch.sum(pab * torch.log((pab + self.smooth_nr) / (papb + self.smooth_dr) + self.smooth_dr), dim=(1, 2))

        if self.reduction == "sum":
            return -torch.sum(mi)
        if self.reduction == "none":
            return -mi
        if self.reduction == "mean":
            return -torch.mean(mi)
        raise ValueError(f"Invalid reduction type: {self.reduction}")



import torch
import torch.nn as nn


class MSELossND(nn.Module):
    """
    Generalized MSE Loss for both 2D and 3D images.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: Specifies the reduction to apply to the output:
                - "none": No reduction.
                - "mean": Mean of all elements.
                - "sum": Sum of all elements.
        """
        super(MSELossND, self).__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted image (2D or 3D).
            target: Target image (2D or 3D). Must have the same shape as `pred`.
        Returns:
            MSE loss value.
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
        loss = (pred - target) ** 2

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

