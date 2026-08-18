from torch.utils.tensorboard import SummaryWriter
import os, glob
import TransMorph.losses as losses
import TransMorph.utils as utils
import sys
from torch.utils.data import DataLoader
from TransMorph.data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from TransMorph.models.TransMorph import CONFIGS as CONFIGS_TM
import TransMorph.models.TransMorph as TransMorph
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R

def get_files(image_root, extend_name):
    image_files_path = os.listdir(image_root)
    path_temp = []
    for i in image_files_path:
        fname, ext = os.path.splitext(i)
        if ext == extend_name:
            i = os.path.join(image_root, i)
            path_temp.append(i)
        pass
    image_files_path = path_temp
    return  image_files_path


def add_noise_and_blur(image, noise_std=0.02, blur_sigma=0.7):
    """
    对图像进行加噪声和模糊处理
    :param image: SimpleITK 图像对象
    :param noise_std: 高斯噪声标准差
    :param blur_sigma: 高斯模糊标准差
    :return: 处理后的 SimpleITK 图像
    """
    img_array = sitk.GetArrayFromImage(image)

    # 加高斯噪声
    noise = np.random.normal(0, noise_std, img_array.shape)
    noisy_img = img_array + noise

    # 加高斯模糊
    blurred_img = gaussian_filter(noisy_img, sigma=blur_sigma)

    # 转回 SimpleITK 图像
    return sitk.GetImageFromArray(blurred_img)


def apply_random_mask(image, mask_prob=0.3, block_size=10):
    """
    在 2D 图像上随机遮盖部分区域。

    :param image: 2D 灰度图像 (numpy 数组)
    :param mask_prob: 遮挡区域的概率（默认 30%）
    :param block_size: 每个遮挡块的大小（默认 10 像素）
    :return: 处理后的 2D 图像
    """
    img_h, img_w = image.shape

    # 生成一个和原图同样大小的遮挡 mask，初始值为 1（不遮挡）
    mask = np.ones((img_h, img_w), dtype=np.float32)

    # 计算要遮挡的区域数量
    num_blocks_h = img_h // block_size
    num_blocks_w = img_w // block_size

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            if np.random.rand() < mask_prob:  # 以一定概率遮挡
                y_start = i * block_size
                x_start = j * block_size
                mask[y_start:y_start + block_size, x_start:x_start + block_size] = 0  # 置零遮挡

    # 应用遮挡
    masked_image = image * mask

    return masked_image

def remove_random_slices(image, removal_prob=0.1):
    """
    对 3D 图像的 Z 轴方向随机挖掉部分数据（设为 0）。

    :param image: SimpleITK 读取的 3D 图像（32-bit 单通道）。
    :param removal_prob: 每个 Z 切片被移除的概率（默认 30%）。
    :return: 处理后的 SimpleITK 图像。
    """
    img_array = sitk.GetArrayFromImage(image)  # 获取 NumPy 数组
    depth = img_array.shape[0]  # 获取 Z 轴方向的切片数

    # 生成一个布尔掩码，决定哪些切片需要被移除
    removal_mask = np.random.rand(depth) < removal_prob

    # 置零被移除的切片
    for ind in range(depth):
        flag = removal_mask[ind]
        if flag:
            temp = img_array[ind, :, :]
            temp = apply_random_mask(temp)
            img_array[ind, :, :] = temp

    # 转换回 SimpleITK 格式
    processed_image = sitk.GetImageFromArray(img_array)
    processed_image.SetSpacing(image.GetSpacing())  # 继承原图的空间信息
    processed_image.SetOrigin(image.GetOrigin())
    processed_image.SetDirection(image.GetDirection())

    return processed_image
def read_img():
    fixed_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\fixed_image"
    moving_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\moving_image"
    fixed_list = get_files(fixed_root, '.nii')[:]
    moving_list = get_files(moving_root, '.nii')[:]

    fixed_label_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\fixed_label"
    moving_label_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\moving_label"
    fixed_label_files = get_files(fixed_label_root, '.nii')[:]
    moving_label_files = get_files(moving_label_root, ".nii")[:]
    save_dir = r"Z:\users\yq\MorphDatasets\Bspine\noise_1114\moving_image"
    os.makedirs(save_dir, exist_ok=True)

    for file_path in moving_list:
        filename = os.path.basename(file_path)
        img = sitk.ReadImage(file_path)
        img = add_noise_and_blur(img)
        img = remove_random_slices(img, removal_prob=0.2)
        save_path = os.path.join(save_dir, filename)

        sitk.WriteImage(img, save_path)
        print(f"Processed and saved: {save_path}")


if __name__ == '__main__':
    read_img()