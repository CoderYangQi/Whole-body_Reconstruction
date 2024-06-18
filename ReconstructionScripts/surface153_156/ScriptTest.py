import unittest

import cv2
import numpy as np
from Step1_Translate import GetOffset

import os
import SimpleITK as sitk
from common_script.ome_tiff import write_ome_tiff

def ReSampleImg(up_path,  upOrigin,  left_point, refSize, spacing):
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
    def test_std(self):
        ImgFormat = r"Z:\Data\E\E-123\Reconstruction\SliceImage\{}_img.tif"
        saveRoot = r"Z:\Data\E\E-123\Reconstruction\saveTemp\th0630_153_155"
        for index in range(153,157):
            img = sitk.ReadImage(ImgFormat.format(index))
            sitk.WriteImage(img[:,:,75],os.path.join(saveRoot,"{}_z75.tif".format(index)))
            sitk.WriteImage(img[:,:,175],os.path.join(saveRoot,"{}_z175.tif".format(index)))
    def test_apply_xy(self):
        save_root = r'Z:\Data\E\E-123\Reconstruction\saveTemp\th0630_153_155'
        prevFormat = r"Z:\Data\E\E-123\Reconstruction\saveTemp\th0630_153_155\2_1_1_{:03d}_561nm_10X_ls.mha"
        nextFormat = r"Z:\Data\E\E-123\Reconstruction\saveTemp\th0630_153_155\2_1_1_{:03d}_561nm_10X_us.mha"
        for index in range(153,156):
            prev_index = index; next_index = index + 1
            save_prev_df = os.path.join(save_root, "2_1_1_{:03d}_561nm_10X_lxy.mha".format(prev_index))
            save_next_df = os.path.join(save_root, "2_1_1_{:03d}_561nm_10X_uxy.mha".format(next_index))
            prev_surface = sitk.ReadImage(prevFormat.format(prev_index))
            next_surface = sitk.ReadImage(nextFormat.format(next_index))
            d1 = sitk.ReadImage(save_prev_df)
            d2 = sitk.ReadImage(save_next_df)
            t1 = sitk.DisplacementFieldTransform(sitk.Image(d1))
            t2 = sitk.DisplacementFieldTransform(sitk.Image(d2))
            out1 = sitk.Resample(prev_surface, d1, t1)
            out2 = sitk.Resample(next_surface, d2, t2)
            save_prev = os.path.join(save_root, "16bit_{:03d}_ls_re.mha".format(prev_index))
            save_next = os.path.join(save_root, "16bit_{:03d}_us_re.mha".format(next_index))
            sitk.WriteImage(out1, save_prev)
            sitk.WriteImage(out2, save_next)

    def test_AlignImgs(self):
        prevImgFormat = r"Z:\Data\E\E-123\Reconstruction\SliceImage\4.0\QIE_0630-1-5_E_{:03d}_648nm_10X.tif"
        nextImgFormat = r"Z:\Data\E\E-123\Reconstruction\SliceImage\4.0\QIE_0630-1-5_E_{:03d}_648nm_10X.tif"
        saveRoot = r"Z:\Data\E\E-123\Reconstruction\saveTemp"

        visorPath = r"Z:\Data\E\E-123\0630test_2.visor"
        leftList, rightList = GetOffset(visorPath)
        leftList = np.array(leftList)
        rightList = np.array(rightList)

        spacing = [1, 1, 1]
        newLeftList = leftList[152:156]
        newRightList = rightList[152:156]
        lefttop = newLeftList.min(axis=0)
        rightbottom = newRightList.max(axis=0)
        lefttop = [lefttop[0], lefttop[1], 0]
        rate = 4
        refSize = [(rightbottom[0] - lefttop[0]) // rate, (rightbottom[1] - lefttop[1]) // rate]
        refSize = [int(i) for i in refSize]
        print(f"refine size: {refSize}")
        for i in range(156, 157):
            prevIndex = i
            nextIndex = i + 1
            upOrigin = leftList[prevIndex - 1]
            # upOrigin = [0,0,0]
            upOrigin[2] = 0
            downOrigin = leftList[nextIndex - 1]
            # downOrigin = [0,0,0]
            downOrigin[2] = 0
            up_path = prevImgFormat.format(prevIndex)
            down_path = nextImgFormat.format(nextIndex)
            # bottom1 = heightPairs[i][1]

            up_img = ReSampleImg(up_path, upOrigin, lefttop, refSize, spacing)
            write_ome_tiff(up_img,r"Z:\Data\E\E-123\Reconstruction\SliceImage\{}_img.tif".format(prevIndex))
    # todo 插值算法
    def test_interpolate(self):
        def combine(tf_xy,tf_z):
            tf_xy = sitk.Compose(sitk.VectorIndexSelectionCast(tf_xy, 0),
                                 sitk.VectorIndexSelectionCast(tf_xy, 1),
                                 sitk.Image(tf_xy.GetSize(), sitk.sitkFloat64))
            # todo tf_z 已经是一个3d 的数据了
            # tf_z = sitk.Compose(sitk.Image(tf_z.GetSize(), sitk.sitkFloat32),
            #                     sitk.Image(tf_z.GetSize(), sitk.sitkFloat32),
            #                     sitk.Cast(tf_z, sitk.sitkFloat32))
            tf_xy = sitk.JoinSeries([tf_xy])
            tf_z = sitk.JoinSeries([tf_z])
            size = tf_xy.GetSize()
            tf1 = sitk.Cast(tf_z, sitk.sitkVectorFloat64)
            tf2 = sitk.Cast(tf_xy, sitk.sitkVectorFloat64)
            df1 = sitk.DisplacementFieldTransform(sitk.Image(tf1))
            df2 = sitk.DisplacementFieldTransform(sitk.Image(tf2))
            tr = sitk.Transform(df1.GetDimension(), sitk.sitkComposite)
            tr.AddTransform(df1)
            tr.AddTransform(df2)
            df = sitk.TransformToDisplacementField(tr, sitk.sitkVectorFloat32, size)
            return df
        imgFormat = r"Z:\Data\E\E-123\Reconstruction\SliceImage\{}_img.tif"
        saveRoot = r"Z:\Data\E\E-123\Reconstruction\saveTemp\th0630_153_155"
        for index in range(156,157):
            # index = 154
    
            uz = sitk.ReadImage(os.path.join(saveRoot, "2_1_1_{:03d}_561nm_10X_uz.mha".format(index)))
            lz = sitk.ReadImage(os.path.join(saveRoot, "2_1_1_{:03d}_561nm_10X_lz.mha".format(index)))

            uz.SetSpacing([1, 1, 1])
            uz.SetOrigin([0, 0, 0])
            lz.SetSpacing([1, 1, 1])
            lz.SetOrigin([0, 0, 0])
            save_udf = os.path.join(saveRoot, "2_1_1_{:03d}_561nm_10X_uxy.mha".format(index))
            save_ldf = os.path.join(saveRoot, "2_1_1_{:03d}_561nm_10X_lxy.mha".format(index))
            uxy = sitk.ReadImage(save_udf)
            lxy = sitk.ReadImage(save_ldf)
            slice_thickness = 100
            roi = [0,0,-400]
            u = combine(uxy,uz)
            l = combine(lxy,lz)
            u = sitk.Compose(sitk.VectorIndexSelectionCast(u, 0) ,
                             sitk.VectorIndexSelectionCast(u, 1) ,
                             sitk.VectorIndexSelectionCast(u, 2)  + (- 0 * slice_thickness))
            l = sitk.Compose(sitk.VectorIndexSelectionCast(l, 0),
                             sitk.VectorIndexSelectionCast(l, 1) ,
                             sitk.VectorIndexSelectionCast(l, 2)  + (- 1 * slice_thickness))
            zero = sitk.Compose(sitk.VectorIndexSelectionCast(u, 0) * 2 - sitk.VectorIndexSelectionCast(l, 0),
                             sitk.VectorIndexSelectionCast(u, 1) * 2 - sitk.VectorIndexSelectionCast(l, 1) ,
                             sitk.Image(u.GetSize(),sitk.sitkFloat32)- 25  + (1 * slice_thickness))
            df = sitk.JoinSeries([zero[:,:,0], u[:, :, 0], l[:, :, 0]])
            # df = sitk.JoinSeries([u[:, :, 0] * 2 - l[:,:,0], u[:, :, 0], l[:, :, 0]])
            df.SetOrigin([0, 0, 0 * slice_thickness])
            df.SetSpacing([1, 1, slice_thickness])
            size = df.GetSize()

            df = sitk.Cast(df, sitk.sitkVectorFloat64)
            df = sitk.DisplacementFieldTransform(df)
            refSize = [size[0], size[1], 200]
            img = sitk.ReadImage(imgFormat.format(index))
            img.SetSpacing([1, 1, 1])
            img.SetOrigin([0, 0, 100])
            sitk.WriteImage(sitk.MaximumProjection(img,projectionDimension=0)[0,:,:], "std_yoz.tif")
            refineImg = sitk.Resample(img, refSize, df,sitk.sitkLinear)
            sitk.WriteImage(refineImg[:,:,100],"result_{}_0.tif".format(index))
            sitk.WriteImage(refineImg[:,:,-1],"result_{}_-1.tif".format(index))
            yoz = sitk.MaximumProjection(refineImg,projectionDimension=0)[0,:,:]
            write_ome_tiff(refineImg,r"D:\USERS\yq\code\heightVISoR\YQReconstructionScripts\surface153_156\resample_{}.tif".format(index))
            sitk.WriteImage(yoz,"yoz.tif")


        # umap_s = umap + 1
        # lmap_s = lmap - 1
        # zeros = sitk.Image(umap.GetSize(), umap.GetPixelIDValue())
        # df = sitk.JoinSeries(sitk.Compose(zeros, zeros, umap_s), sitk.Compose(zeros, zeros, lmap_s))
    def test_combine(self):
        root = r"Z:\Data\E\E-123\Reconstruction\saveTemp\th0630_153_155\result"
        lsFormat = os.path.join(root,"2_1_1_{:03d}_561nm_10X_ls.mha")
        usFormat = os.path.join(root,"2_1_1_{:03d}_561nm_10X_us.mha")
        saveSFormat = os.path.join(root,"coarse_{:03d}_{:03d}.tif")
        for index in range(153,156):
            prevIndex = index
            nextIndex = index + 1
            ls = sitk.ReadImage(lsFormat.format(prevIndex))
            us = sitk.ReadImage(usFormat.format(nextIndex))
            CombineChannel(ls,us,saveSFormat.format(prevIndex,nextIndex))
        print()

        std75Format = os.path.join(root, "{}_z75.tif")
        std175Format = os.path.join(root, "{}_z175.tif")
        savestdFormat = os.path.join(root, "std_{:03d}_{:03d}.tif")
        for index in range(153, 156):
            prevIndex = index
            nextIndex = index + 1
            std175 = sitk.ReadImage(std175Format.format(prevIndex))
            std75 = sitk.ReadImage(std75Format.format(nextIndex))
            CombineChannel(std175, std75, savestdFormat.format(prevIndex, nextIndex))
        print()
        RelsFormat = os.path.join(root, "16bit_{}_ls_re.mha")
        ReusFormat = os.path.join(root, "16bit_{}_us_re.mha")
        saveReFormat = os.path.join(root, "Re_{:03d}_{:03d}.tif")
        for index in range(153, 156):
            prevIndex = index
            nextIndex = index + 1
            rels = sitk.ReadImage(RelsFormat.format(prevIndex))
            reus = sitk.ReadImage(ReusFormat.format(nextIndex))
            CombineChannel(rels, reus, saveReFormat.format(prevIndex, nextIndex))
        print()
    def test_get_roi(self):
        img_path = r"D:\USERS\yq\code\heightVISoR\YQReconstructionScripts\surface153_156\resample_{}.tif"
        index = 153
        flagProcess = False
        prevIndex = index
        nextIndex = index +  1
        # img1 = sitk.ReadImage(img_path.format(prevIndex))
        # img2 = sitk.ReadImage(img_path.format(nextIndex))
        # roi = [[1500,2114],[2100,2216]] # _0617_2
        # roi1 = img1[roi[0][0]:roi[1][0],roi[0][1]:roi[1][1],:]
        # roi2 = img2[roi[0][0]:roi[1][0],roi[0][1]:roi[1][1],:]
        # sitk.WriteImage(roi1,r"roi_{}_0617_2.tif".format(prevIndex))
        # sitk.WriteImage(roi2,r"roi_{}_0617_2.tif".format(nextIndex))

        roi1 = sitk.ReadImage(r"roi_{}_0617_2.tif".format(prevIndex))
        roi2 = sitk.ReadImage(r"roi_{}_0617_2.tif".format(nextIndex))
        y_range = [55, 58]
        zox1 = sitk.MaximumProjection(roi1[:, y_range[0]:y_range[1], :], projectionDimension=1)[:, 0, :]
        zox2 = sitk.MaximumProjection(roi2[:, y_range[0]:y_range[1], :], projectionDimension=1)[:, 0, :]
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

        img1_resize = sitk.GetImageFromArray(img1_resize)
        img2_resize = sitk.GetImageFromArray(img2_resize)
        if flagProcess:
            threshold = 150
            img1_resize = Preprocess(img1_resize,threshold)
            img2_resize = Preprocess(img2_resize,threshold)

        CombineChannel(img1_resize, img2_resize, "combine_method_roi.tif")
    def test_getstd_roi(self):
        img_path = r"Z:\Data\E\E-123\Reconstruction\SliceImage\{}_img.tif"
        index = 153
        prevIndex = index
        nextIndex = index +  1
        # img1 = sitk.ReadImage(img_path.format(prevIndex))
        # img2 = sitk.ReadImage(img_path.format(nextIndex))
        # roi = [[1500,2114],[2100,2216]] # _0617_2
        # roi1 = img1[roi[0][0]:roi[1][0],roi[0][1]:roi[1][1],:175]
        # roi2 = img2[roi[0][0]:roi[1][0],roi[0][1]:roi[1][1],:175]
        # sitk.WriteImage(roi1,r"std_roi_{}_0617_2.tif".format(prevIndex))
        # sitk.WriteImage(roi2,r"std_roi_{}_0617_2.tif".format(nextIndex))

        roi1 = sitk.ReadImage(r"std_roi_{}_0617_2.tif".format(prevIndex))
        roi2 = sitk.ReadImage(r"std_roi_{}_0617_2.tif".format(nextIndex))

        y_range = [55, 58]
        zox1 = sitk.MaximumProjection(roi1[:, y_range[0]:y_range[1], :], projectionDimension=1)[:, 0, :]
        zox2 = sitk.MaximumProjection(roi2[:, y_range[0]:y_range[1], :], projectionDimension=1)[:, 0, :]
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

        img1_resize = img1_resize[:,185:]
        img2_resize = img2_resize[:,185:]
        img2_resize[:130,:] = 0
        CombineChannel(sitk.GetImageFromArray(img1_resize), sitk.GetImageFromArray(img2_resize), "combine_std_roi.tif")
    def test_combine_roi(self):
        print()
        roi1 = cv2.imread(r'std_roi_20um_zox_prev.tif',-1)
        roi2 = cv2.imread('std_roi_20um_zox_next.tif',-1)
        size1 = roi1.shape
        size2 = roi2.shape
        combineY = size2[0] + 100
        img1_resize = np.zeros([combineY,size2[1]], dtype=np.uint16)
        img2_resize = np.zeros([combineY,size2[1]], dtype=np.uint16)
        img1_resize[:size1[0],:] = roi1
        img2_resize[100:,] = roi2

        CombineChannel(sitk.GetImageFromArray(img1_resize), sitk.GetImageFromArray(img2_resize),"combine_std_roi.tif")

def Preprocess(surface,threshold):
    # if img_path == None:
    #     return None
    # surface = sitk.ReadImage(img_path)
    # threshold = 120
    surface = sitk.Threshold(surface, threshold, 65535, threshold)
    back_log_value = np.log(threshold)
    # back_log_value = 0
    surface = sitk.Clamp((sitk.Log(sitk.Cast(surface + 1, sitk.sitkFloat32)) - back_log_value) * 39.4,
                         sitk.sitkFloat32, 0, 255)


    return surface
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