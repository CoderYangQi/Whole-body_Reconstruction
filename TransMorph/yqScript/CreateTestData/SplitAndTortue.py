import pathlib
import SimpleITK as sitk
from TransMorph.yqScript.ome_tiff import write_ome_tiff
import numpy as np
import time, json
import unittest,os


# prev img 需要提前划分好数据的厚度
def SpiltData(prev_img, img_save_path_format, block_size=[256, 256],interval = 56):
    back_brightness = 120
    rate_threshold = 0.4
    origin = [0, 0, 0]
    prev_img = PreProcess(prev_img)
    img_size = prev_img.GetSize()
    step = block_size[0] - interval
    xy_times = [int(np.floor((img_size[0] - interval) / step))
        , int(np.floor((img_size[1] - interval) / step))]
    pathDict = {}
    ct = 0
    for i in range(xy_times[0]):
        for j in range(xy_times[1]):
            xRange = [i * step, i * step + block_size[0]]
            yRange = [j * step, j * step + block_size[1]]
            prev_save_path = img_save_path_format.format(ct)
            ct += 1
            prev_temp = prev_img[xRange[0]:xRange[1], yRange[0]:yRange[1],:]
            pathDict[f"{i}_{j}"] = prev_save_path
            sitk.WriteImage(prev_temp, prev_save_path)
    return pathDict
def PreProcess(image):
    image = sitk.Cast(image, sitk.sitkFloat32)
    image = sitk.Clamp((sitk.Log(sitk.Cast(image, sitk.sitkFloat32)) - 4.6) * 39.4, sitk.sitkFloat32, 0,
                       255)
    return image
def BSpineTransform(image):
    # 设置 B-spline 变换的参数
    transformDomainMeshSize = [20, 20, 4]  # 控制点网格的尺寸
    bsplineTransform = sitk.BSplineTransformInitializer(image, transformDomainMeshSize)

    # 为 B-spline 变换生成随机偏移
    params = np.array(bsplineTransform.GetParameters(), dtype=np.float64)
    params = params + np.random.randn(params.shape[0]) * 10  # 添加随机偏移，幅度为 10

    # 应用随机偏移到 B-spline 变换
    bsplineTransform.SetParameters(tuple(params))

    # 应用形变到图像
    deformed_image = sitk.Resample(image, bsplineTransform, sitk.sitkLinear, 0.0, image.GetPixelID())
    # deformed_label = sitk.Resample(label, bsplineTransform, sitk.sitkLinear, 0.0, image.GetPixelID())

    # 计算并保存位移场
    displacement_field = sitk.TransformToDisplacementField(bsplineTransform,
                                                           sitk.sitkVectorFloat64,
                                                           image.GetSize(),
                                                           image.GetOrigin(),
                                                           image.GetSpacing(),
                                                           image.GetDirection())

    # 转换回 numpy 数组（如果需要）
    # deformed_array = sitk.GetArrayFromImage(deformed_image)
    # displacement_field_array = sitk.GetArrayFromImage(displacement_field)

    return deformed_image, displacement_field
    # sitk.WriteImage(deformed_label,saveLabelPath)

class TestSplitAndTortue(unittest.TestCase):
    def test_print(self):
        print("Testing SplitAndTortue")
    def test_readImage(self):
        imgPath = r"Z:\Data\E\E-123\Reconstruction\SliceImage\4.0\QIE_0630-1-5_E_155_648nm_10X.tif"
        saveRoot = r"Z:\users\yq\MorphDatasets\Bspine\0811"
        saveImagePath = os.path.join(saveRoot, "testBSpine.tif")
        saveDFpath = os.path.join(saveRoot, "testDF.mha")
        start = [2500,1300]
        img = sitk.ReadImage(imgPath)[start[0]: start[0] + 1000, start[1]:start[1]+1000, -110:-10]
        img.SetOrigin([0,0,0])
        img.SetSpacing([1,1,1])
        sitk.WriteImage(img, os.path.join(saveRoot, 'testOrigin.tif'))
        # write_ome_tiff(img,os.path.join(saveRoot, 'testOrigin.tif'))
        deformed_image, displacement_field = BSpineTransform(img)
        # 保存位移场和形变后的图像
        sitk.WriteImage(deformed_image, saveImagePath)
        sitk.WriteImage(displacement_field, saveDFpath)

        basename = os.path.basename(imgPath).split('.')[0]
        print(basename)
        originRoot = os.path.join(saveRoot, 'fixed')
        if not os.path.exists(originRoot):
            os.mkdir(originRoot)
        nameFormat = os.path.join(originRoot, basename + "_{}_{}.tif")
        SpiltData(img, nameFormat,interval=50)

        # save DF
        BRoot = os.path.join(saveRoot, 'moving')
        if not os.path.exists(BRoot):
            os.mkdir(BRoot)

        BnameFormat = os.path.join(BRoot, basename + "_{}_{}.tif")
        SpiltData(deformed_image, BnameFormat, interval=50)
    def test_readImage(self):
        imgPath = r"Z:\Data\E\E-123\Reconstruction\SliceImage\4.0\QIE_0630-1-5_E_155_648nm_10X.tif"
        basename = os.path.basename(imgPath).split('.')[0]
        saveRoot = r"Z:\users\yq\MorphDatasets\Bspine\0811"
        saveImagePath = os.path.join(saveRoot, "testBSpine.tiff")
        saveDFpath = os.path.join(saveRoot, "testDF.mha")
        img = sitk.ReadImage(os.path.join(saveRoot, 'testOrigin.tif'))
        img.SetOrigin([0,0,0])
        img.SetSpacing([1,1,1])
        # sitk.WriteImage(img, os.path.join(saveRoot, 'testOrigin.tif'))
        # write_ome_tiff(img,os.path.join(saveRoot, 'testOrigin.tif'))
        # deformed_image, displacement_field = BSpineTransform(img)
        deformed_image = sitk.ReadImage(saveImagePath)
        # 保存位移场和形变后的图像
        # sitk.WriteImage(deformed_image, saveImagePath)

        originRoot = os.path.join(saveRoot, 'fixed')
        if not os.path.exists(originRoot):
            os.mkdir(originRoot)
        # nameFormat = os.path.join(originRoot, basename + "_{}_{}.tif")
        nameFormat = (originRoot+ '\\{:04d}.tif')
        pathDict = SpiltData(img, nameFormat,interval=50)
        # 将字典保存为 JSON 文件
        file_path = os.path.join(saveRoot, "fixed_names.json")
        with open(file_path, 'w') as json_file:
            json.dump(pathDict, json_file, indent=4, ensure_ascii=False)

        # save DF
        BRoot = os.path.join(saveRoot, 'moving')
        if not os.path.exists(BRoot):
            os.mkdir(BRoot)

        # BnameFormat = os.path.join(BRoot, "{:04d}.tif")
        BnameFormat = (BRoot+"\\{:04d}.tif")
        pathDict = SpiltData(deformed_image, BnameFormat, interval=50)
        file_path = os.path.join(saveRoot, "moving_names.json")
        with open(file_path, 'w') as json_file:
            json.dump(pathDict, json_file, indent=4, ensure_ascii=False)