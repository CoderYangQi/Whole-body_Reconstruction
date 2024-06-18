import unittest

import cv2
import numpy as np

import SimpleITK as sitk


def ReSampleImg(up_path, upOrigin, left_point, refSize, spacing):
    print(f"Reconstruction started for data chunk ")
    print(f"String input: {up_path}")

    # print(f"Reconstruction completed for data chunk {data_id}")
    up_img = sitk.ReadImage(up_path)

    # todo 不需要做全局的 填充，仅仅只用在意邻近片之间的问题
    up_img.SetOrigin(upOrigin)
    up_img.SetSpacing(spacing)
    # init transform
    dimension = 3
    up_size = up_img.GetSize()
    up_img = sitk.Resample(up_img, [refSize[0], refSize[1], up_size[2]],
                           sitk.Transform(), sitk.sitkLinear, left_point, spacing)

    # 统一 数据的大小范围
    # todo 默认不做 扩充，但是可能造成数据的缺失
    # left_point = [0,0,0]
    print("left_point is : ", left_point)
    # todo

    # sitk.WriteImage()

    print("up_img.GetSpacing() : {}\n Origin: {}\n Size: {}".format(up_img.GetSpacing(), up_img.GetOrigin(),
                                                                    up_img.GetSize()))
    return up_img


class Calloss(unittest.TestCase):
    def test_print(self):
        print("start")

    # # roi = [[6938,3691],[7298,3768]] # _0616_2
    # roi = [[866,1104],[1413,1312]] # _0616_3
    def test_get_roi(self):
        img_path = r"D:\USERS\yq\code\heightVISoR\YQReconstructionScripts\surface153_156\resample_{}.tif"
        index = 154
        prevIndex = index
        nextIndex = index + 1
        # img1 = sitk.ReadImage(img_path.format(prevIndex))
        # img2 = sitk.ReadImage(img_path.format(nextIndex))
        # roi = [[1470,2120],[2084,2281]] # _0617_1
        # roi1 = img1[roi[0][0]:roi[1][0],roi[0][1]:roi[1][1],:]
        # roi2 = img2[roi[0][0]:roi[1][0],roi[0][1]:roi[1][1],:]
        # sitk.WriteImage(roi1,r"roi_{}_0617_1.tif".format(prevIndex))
        # sitk.WriteImage(roi2,r"roi_{}_0617_1.tif".format(nextIndex))

        roi1 = sitk.ReadImage(r"roi_{}_0617_1.tif".format(prevIndex))
        roi2 = sitk.ReadImage(r"roi_{}_0617_1.tif".format(nextIndex))

        zox1 = sitk.MaximumProjection(roi1[:, 10:20, :], projectionDimension=1)[:, 0, :]
        zox2 = sitk.MaximumProjection(roi2[:, 10:20, :], projectionDimension=1)[:, 0, :]
        sitk.WriteImage(zox1, 'roi_20um_zox_prev.tif')
        sitk.WriteImage(zox2, 'roi_20um_zox_next.tif')

        # combine
        roi1 = cv2.imread(r'roi_20um_zox_prev.tif', -1)
        roi2 = cv2.imread('roi_20um_zox_next.tif', -1)
        size1 = roi1.shape
        size2 = roi2.shape
        combineY = size2[0] + 100
        img1_resize = np.zeros([combineY, size2[1]], dtype=np.uint16)
        img2_resize = np.zeros([combineY, size2[1]], dtype=np.uint16)
        img1_resize[:size1[0], :] = roi1
        img2_resize[100:, ] = roi2

        # img1_resize = img1_resize[:, 185:]
        # img2_resize = img2_resize[:, 185:]
        # img2_resize[:155, :] = 0

        CombineChannel(sitk.GetImageFromArray(img1_resize), sitk.GetImageFromArray(img2_resize),
                       "combine_method_roi.tif")

    def test_getstd_roi(self):
        img_path = r"Z:\Data\E\E-123\Reconstruction\SliceImage\{}_img.tif"
        index = 154
        prevIndex = index
        nextIndex = index + 1
        # img1 = sitk.ReadImage(img_path.format(prevIndex))
        # img2 = sitk.ReadImage(img_path.format(nextIndex))
        # roi = [[1470,2120],[2084,2281]] # _0617_1
        # roi1 = img1[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1], :175]
        # roi2 = img2[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1], :175]
        # sitk.WriteImage(roi1, r"std_roi_{}_0617_1.tif".format(prevIndex))
        # sitk.WriteImage(roi2, r"std_roi_{}_0617_1.tif".format(nextIndex))

        roi1 = sitk.ReadImage(r"std_roi_{}_0617_1.tif".format(prevIndex))
        roi2 = sitk.ReadImage(r"std_roi_{}_0617_1.tif".format(nextIndex))

        zox1 = sitk.MaximumProjection(roi1[:, 0:10, :], projectionDimension=1)[:, 0, :]
        zox2 = sitk.MaximumProjection(roi2[:, 0:10, :], projectionDimension=1)[:, 0, :]
        sitk.WriteImage(zox1, 'std_roi_20um_zox_prev.tif')
        sitk.WriteImage(zox2, 'std_roi_20um_zox_next.tif')

        # combine img
        roi1 = cv2.imread('std_roi_20um_zox_prev.tif', -1)
        roi2 = cv2.imread('std_roi_20um_zox_next.tif', -1)
        size1 = roi1.shape
        size2 = roi2.shape
        combineY = size2[0] + 100
        img1_resize = np.zeros([combineY, size2[1]], dtype=np.uint16)
        img2_resize = np.zeros([combineY, size2[1]], dtype=np.uint16)
        img1_resize[:size1[0], :] = roi1
        img2_resize[100:, ] = roi2

        # img1_resize = img1_resize[:,185:]
        # img2_resize = img2_resize[:,185:]
        # img2_resize[:130,:] = 0
        CombineChannel(sitk.GetImageFromArray(img1_resize), sitk.GetImageFromArray(img2_resize), "combine_std_roi.tif")

    def test_combine_roi(self):
        print()
        roi1 = cv2.imread(r'std_roi_20um_zox_prev.tif', -1)
        roi2 = cv2.imread('std_roi_20um_zox_next.tif', -1)
        size1 = roi1.shape
        size2 = roi2.shape
        combineY = size2[0] + 100
        img1_resize = np.zeros([combineY, size2[1]], dtype=np.uint16)
        img2_resize = np.zeros([combineY, size2[1]], dtype=np.uint16)
        img1_resize[:size1[0], :] = roi1
        img2_resize[100:, ] = roi2

        CombineChannel(sitk.GetImageFromArray(img1_resize), sitk.GetImageFromArray(img2_resize), "combine_std_roi.tif")


def Preprocess(surface, threshold):
    # if img_path == None:
    #     return None
    # surface = sitk.ReadImage(img_path)
    # threshold = 120
    surface = sitk.Threshold(surface, threshold, 65535, threshold)
    back_log_value = np.log(threshold)
    # back_log_value = 0
    surface = sitk.Clamp((sitk.Log(sitk.Cast(surface + 1, sitk.sitkFloat32)) - back_log_value) * 39.4,
                         sitk.sitkFloat32, 0, 255)


def CombineChannel(img1, img2, output_path):
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