import argparse
import json
import re
import unittest
import os
import SimpleITK as sitk
from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
def mainCreateXY():
    root = r"D:\USERS\yq\code\heightVISoR\YQReconstructionScripts\surface153_156"
    uxyFormat = os.path.join(root,r"2_1_1_{:03d}_561nm_10X_uxy.mha")
    lxyFormat = os.path.join(root,r"2_1_1_{:03d}_561nm_10X_lxy.mha")
    uzFormat = os.path.join(root,r"2_1_1_{:03d}_561nm_10X_uz.mha")
    lzFormat = os.path.join(root,r"2_1_1_{:03d}_561nm_10X_lz.mha")
    img_size =  [8375, 4500]
    for sliceIndex in range(156,157):
        print(f"sliceIndex is {sliceIndex}")
        uxy = sitk.Image()

        umap_z = sitk.Image([img_size[0], img_size[1]], sitk.sitkFloat64) + 75

        umap_y = sitk.Image([img_size[0], img_size[1]], sitk.sitkFloat64);
        umap_x = sitk.Image([img_size[0], img_size[1]], sitk.sitkFloat64)

        lmap_z = sitk.Image([img_size[0], img_size[1]], sitk.sitkFloat64) + 175

        lmap_y = sitk.Image([img_size[0], img_size[1]], sitk.sitkFloat64);
        lmap_x = sitk.Image([img_size[0], img_size[1]], sitk.sitkFloat64)
        # todo  使用 zero 替代 x和y的位移

        # todo 保存 uz lz mha 不同于 之前的 uz lz 此处的数据是有 x y z 三个维度的形变场

        uz = sitk.Compose(umap_x, umap_y, umap_z)
        lz = sitk.Compose(lmap_x, lmap_y, lmap_z)
        uxy = sitk.Compose(umap_x, umap_y)
        lxy = sitk.Compose(lmap_x,lmap_y)
        sitk.WriteImage(uxy,uxyFormat.format(sliceIndex))
        sitk.WriteImage(lxy, lxyFormat.format(sliceIndex))

    pass
# todo 将 位移场 添加offset到BrainTransform中


if __name__ == '__main__':
    mainCreateXY()
