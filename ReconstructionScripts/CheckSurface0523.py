
import unittest
import SimpleITK as sitk
import os
import numpy as np
import time
from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
def CombineChannel(img1,img2,output_path):
    # 将SimpleITK图像转换为NumPy数组
    channel1_array = sitk.GetArrayFromImage(img1)
    channel2_array = sitk.GetArrayFromImage(img2)

    # 创建一个新的数组来存储多通道数据
    # 在numpy中，形状应该是 (depth, height, width, channels)
    multi_channel_array = np.stack((channel1_array, channel2_array), axis=-1)

    # 检查新数组的形状
    print(f'Multi-channel image shape: {multi_channel_array.shape}')

    # # 保存为多通道TIF格式文件
    # output_tif_path = saveFormat.format(prevIndex, nextIndex)
    # tiff.imwrite(output_tif_path, multi_channel_array.astype('float32'))

    # 将NumPy数组转换回SimpleITK图像
    multi_channel_image = sitk.GetImageFromArray(multi_channel_array, isVector=True)

    # 设置图像的元数据（例如方向、间距和原点）
    multi_channel_image.SetDirection(img1.GetDirection())
    multi_channel_image.SetSpacing(img1.GetSpacing())
    multi_channel_image.SetOrigin(img1.GetOrigin())

    # 保存多通道图像
    # output_path = saveFormat.format(prevIndex, nextIndex)

    sitk.WriteImage(multi_channel_image, output_path)


class TestAddFunction(unittest.TestCase):
    def test_print(self):
        print("test")

    def test_CheckSurface(self):
        TempRoot = r"D:\USERS\yq\TH2_Part\90_110\Reconstruction\Temp"
        lxyFormat = os.path.join(TempRoot, "2_1_1_{:03d}_561nm_10X_lxy.mha")
        uxyFormat = os.path.join(TempRoot, "2_1_1_{:03d}_561nm_10X_uxy.mha")
        lsFormat = os.path.join(TempRoot, "2_1_1_{:03d}_561nm_10X_ls.mha")
        usFormat = os.path.join(TempRoot, "2_1_1_{:03d}_561nm_10X_us.mha")
        saveRoot = TempRoot
        savelsFormat = saveRoot + "/{:03d}ls.mha"
        saveusFormat = saveRoot + "/{:03d}us.mha"
        saveChannelFormat = saveRoot + "/{:03d}_{:03d}_channel.tif"
        for index in range(78, 100):
            prev_index = index
            next_index = index + 1
            # 2_1_1_{:03d}_561nm_10X
            d1 = sitk.ReadImage(lxyFormat.format(prev_index))
            d2 = sitk.ReadImage(uxyFormat.format(next_index))
            prev_surface = sitk.ReadImage(lsFormat.format(prev_index))
            next_surface = sitk.ReadImage(usFormat.format(next_index))

            t1 = sitk.DisplacementFieldTransform(sitk.Image(d1))
            t2 = sitk.DisplacementFieldTransform(sitk.Image(d2))
            out1 = sitk.Resample(prev_surface, d1, t1)
            out2 = sitk.Resample(next_surface, d2, t2)
            print(f"prev index is {prev_index}")
            sitk.WriteImage(out1, savelsFormat.format(prev_index))
            sitk.WriteImage(out2, saveusFormat.format(next_index))
            output_path = saveChannelFormat.format(prev_index, next_index)
            CombineChannel(out1, out2, output_path)
    def test_alignSurface(self):
        save_root = r'D:\USERS\yq\TH2_Part\155_185\Reconstruction\Temp'
        ls_name_format = "2_1_1_{:03d}_561nm_10X_ls"
        root = r'D:\USERS\yq\TH2_Part\155_185\Reconstruction\Temp'
        prevFormat = os.path.join(root, '2_1_1_{:03d}_561nm_10X_ls.mha')
        nextFormat = os.path.join(root, '2_1_1_{:03d}_561nm_10X_us.mha')
        pointsFlag = False
        printMsg = True
        for i in range(179, 185):
            start = time.time()
            prev_index = i
            next_index = i + 1
            if pointsFlag:
                prev_points = r"D:\USERS\yq\TH2_Reconstruction\Annotation\SurfaceRegistration\{:03d}_lp.txt".format(
                    prev_index)
                next_points = r"D:\USERS\yq\TH2_Reconstruction\Annotation\SurfaceRegistration\{:03d}_up.txt".format(
                    next_index)
            else:
                prev_points = None;
                next_points = None
            prev_path = prevFormat.format(prev_index)
            next_path = nextFormat.format(next_index)
            save_prev = os.path.join(save_root, "{:03d}_ls_re.mha".format(prev_index))
            save_next = os.path.join(save_root, "{:03d}_us_re.mha".format(next_index))
            save_prev_2 = os.path.join(save_root, "2_{:03d}_ls_re.mha".format(prev_index))
            save_next_2 = os.path.join(save_root, "2_{:03d}_us_re.mha".format(next_index))

            save_prev_df = os.path.join(save_root, "{:03d}_lxy.mha".format(prev_index))
            save_next_df = os.path.join(save_root, "{:03d}_uxy.mha".format(next_index))
            if printMsg:
                print(f"prev_path is {prev_path}")
                print(f"next_path is {next_path}")
                print(f"prev_points is {prev_points}")
                print(f"next_points is {next_points}")
                print(f"save_prev is {save_prev}")
                print(f"save_next is {save_next}")
                print(f"save_prev_df is {save_prev_df}")
                print(f"save_next_df is {save_next_df}")

            rate = 1

            prev_surface = sitk.ReadImage(prev_path)[::rate, ::rate]

            next_surface = sitk.ReadImage(next_path)[::rate, ::rate]
            ref_size = prev_surface.GetSize()
            prev_surface.SetSpacing([1, 1])
            prev_surface.SetOrigin([0, 0])
            next_surface.SetSpacing([1, 1])
            next_surface.SetOrigin([0, 0])
            # d1, d2, _prev_surface, _next_surface = align_surfaces(prev_surface=prev_surface, next_surface=next_surface, method='yqRefine_elasitx',
            #                         ref_img=None,
            #                         outside_brightness=2, ref_scale=1, ref_size=ref_size, prev_points=prev_points,
            #                         next_points=next_points)
            d1, d2 = align_surfaces(prev_surface=prev_surface, next_surface=next_surface,
                                    method='yqRefine_elastix0523',
                                    ref_img=None,
                                    outside_brightness=2, ref_scale=1, ref_size=ref_size,
                                    prev_points=prev_points,
                                    next_points=next_points)
            print(f"finished time is {time.time() - start}")
            # sitk.WriteImage(sitk.DisplacementFieldJacobianDeterminant(d1),
            #                 save_prev_df)
            # sitk.WriteImage(sitk.DisplacementFieldJacobianDeterminant(d2),
            #                 save_next_df)
            sitk.WriteImage(d1, save_prev_df)
            sitk.WriteImage(d2, save_next_df)
            t1 = sitk.DisplacementFieldTransform(sitk.Image(d1))
            t2 = sitk.DisplacementFieldTransform(sitk.Image(d2))
            out1 = sitk.Resample(prev_surface, d1, t1)
            out2 = sitk.Resample(next_surface, d2, t2)
            sitk.WriteImage(out1, save_prev)
            sitk.WriteImage(out2, save_next)


# if __name__ == '__main__':
#     CheckSurface()