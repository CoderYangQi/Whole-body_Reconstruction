import json

import numpy as np
from scipy.optimize import minimize
import unittest,os
import SimpleITK as sitk
import TransMorph.losses
import TransMorph.utils as utils
from torch.utils.data import DataLoader
from TransMorph.data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from TransMorph.models.TransMorph import CONFIGS as CONFIGS_TM
import TransMorph.models.TransMorph as TransMorph


def useDF(x_path, flow_path, save_path):
    x = sitk.ReadImage(x_path)
    x = sitk.GetArrayFromImage(x)
    x = torch.from_numpy(x).unsqueeze_(0).unsqueeze_(0)
    if flow_path.endswith('.npy'):
        flow = np.load(flow_path)
    elif flow_path.endswith('npz'):
        flow = np.load(flow_path)
        flow = flow['arr_0']
    else:
        print(f"{flow_path} is wrong")
        return None

    flow = torch.from_numpy(flow).unsqueeze_(0).float()
    H, W, D = 64, 256, 256
    config = CONFIGS_TM['TransMorph']
    config.img_size = (H, W, D)
    config.window_size = (H // 16, W // 32, D // 32)
    model = TransMorph.TransMorph(config)
    # debug my flow test
    out = model.spatial_trans(x, flow)
    out = np.squeeze(out.detach().cpu().numpy())
    sitk.WriteImage(sitk.GetImageFromArray(out),
                    save_path)

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


def resize_image_z_center(image, start_z, end_z, target_depth=64):
    # 提取指定的z方向范围
    cropped_image = image[:, :, start_z:end_z]

    # 将提取的图像转换为NumPy数组
    cropped_array = sitk.GetArrayFromImage(cropped_image)

    # 当前图像深度
    current_depth = cropped_array.shape[0]

    # 如果当前深度小于目标深度，需要进行填充
    if current_depth < target_depth:
        # 计算需要填充的上下边界
        padding_before = (target_depth - current_depth) // 2
        padding_after = target_depth - current_depth - padding_before

        # 在z轴方向填充0值，使其达到目标深度
        padded_array = np.pad(cropped_array, ((padding_before, padding_after), (0, 0), (0, 0)), mode='constant',
                              constant_values=0)
    else:
        # 如果当前深度大于或等于目标深度，则进行裁剪
        start_idx = (current_depth - target_depth) // 2
        end_idx = start_idx + target_depth
        padded_array = cropped_array[start_idx:end_idx, :, :]

    # 将数组转换回SimpleITK图像
    resized_image = sitk.GetImageFromArray(padded_array)
    # resized_image.CopyInformation(image)  # 保留原图像的坐标信息
    return resized_image
def PreProcess(image):
    image = sitk.Cast(image, sitk.sitkFloat32)
    image = sitk.Clamp((sitk.Log(sitk.Cast(image, sitk.sitkFloat32)) - 4.6) * 39.4, sitk.sitkFloat32, 0,
                       255)
    return image

class TestModel2DF(unittest.TestCase):
    def test_print(self):
        print("start")

    def test_maxProjection(self):
        originRoot = r"Z:\users\yq\MorphDatasets\OriginDatasets"
        prevImage = sitk.ReadImage(os.path.join(originRoot,"QIE_0630-1-5_E_053_648nm_10X.tif"))
        nextImage = sitk.ReadImage(os.path.join(originRoot,"QIE_0630-1-5_E_054_648nm_10X.tif"))
        yMaxprev = sitk.MaximumProjection(prevImage,projectionDimension=1)[:,0,:]
        yMaxnext = sitk.MaximumProjection(nextImage,projectionDimension=1)[:,0,:]
        sitk.WriteImage(yMaxprev[:,:185],os.path.join(originRoot,"yMaxprev.tif"))
        sitk.WriteImage(yMaxnext[:7409,:185],os.path.join(originRoot,"yMaxnext.tif"))


    def test_getGoodOverlap(self):
        originRoot = r"Z:\users\yq\MorphDatasets\OriginDatasets"
        prevImage = sitk.ReadImage(os.path.join(originRoot, "QIE_0630-1-5_E_053_648nm_10X.tif"))
        nextImage = sitk.ReadImage(os.path.join(originRoot, "QIE_0630-1-5_E_054_648nm_10X.tif"))
        prevImage = prevImage[:, :, 186 - 75:186]
        nextImage = nextImage[:7409, :4033, :75]
        sitk.WriteImage(prevImage, os.path.join(originRoot, f"prev_{186 - 75}_{186}.tif"))
        sitk.WriteImage(nextImage, os.path.join(originRoot, f"next_{0}_{75}.tif"))

        zMaxprev = sitk.MaximumProjection(prevImage, projectionDimension=2)[:, :, 0]
        zMaxnext = sitk.MaximumProjection(nextImage, projectionDimension=2)[:, :, 0]

        sitk.WriteImage(zMaxprev, os.path.join(originRoot, "zMaxprev.tif"))
        sitk.WriteImage(zMaxnext, os.path.join(originRoot, "zMaxnext.tif"))

    def test_split_overlap(self):
        range_ = [[3000,2000],[3000 + 2000,2000 + 2000]]

        originRoot = r"Z:\users\yq\MorphDatasets\OriginDatasets"
        prevImage = sitk.ReadImage(os.path.join(originRoot,\
                   f"prev_{186 - 75}_{186}.tif"))[range_[0][0]: range_[1][0], range_[0][1]: range_[1][1]]
        nextImage = sitk.ReadImage(os.path.join(originRoot,\
                    f"next_{0}_{75}.tif"))[range_[0][0]: range_[1][0], range_[0][1]: range_[1][1]]
        saveRoot = r"Z:\users\yq\MorphDatasets\model\TransMorph\1101\TestResult"
        if not os.path.exists(saveRoot):
            os.makedirs(saveRoot)
        prevImage.SetOrigin([0,0,0])
        prevImage.SetSpacing([1,1,1])
        nextImage.SetOrigin([0,0,0])
        nextImage.SetSpacing([1,1,1])

        #todo set the useful into the middle position

        # 假设 prevImage 是 SimpleITK 图像
        start_z = 75 - 40
        end_z = 75
        prevImage = resize_image_z_center(prevImage, start_z, end_z)
        nextImage = resize_image_z_center(nextImage, start_z, end_z)

        prevImage.SetOrigin([0, 0, 0])
        prevImage.SetSpacing([1, 1, 1])
        nextImage.SetOrigin([0, 0, 0])
        nextImage.SetSpacing([1, 1, 1])
        sitk.WriteImage(prevImage,"resized_prev_image.tif")
        sitk.WriteImage(nextImage,"resized_next_image.tif")
        originRoot = os.path.join(saveRoot, 'fixed')
        if not os.path.exists(originRoot):
            os.mkdir(originRoot)
        # nameFormat = os.path.join(originRoot, basename + "_{}_{}.tif")
        nameFormat = (originRoot + '\\{:04d}.tif')
        pathDict = SpiltData(prevImage, nameFormat, interval=50)
        # 将字典保存为 JSON 文件
        file_path = os.path.join(saveRoot, "fixed_names.json")
        with open(file_path, 'w') as json_file:
            json.dump(pathDict, json_file, indent=4, ensure_ascii=False)

        # save DF
        BRoot = os.path.join(saveRoot, 'moving')
        if not os.path.exists(BRoot):
            os.mkdir(BRoot)

        # BnameFormat = os.path.join(BRoot, "{:04d}.tif")
        BnameFormat = (BRoot + "\\{:04d}.tif")
        pathDict = SpiltData(nextImage, BnameFormat, interval=50)
        file_path = os.path.join(saveRoot, "moving_names.json")
        with open(file_path, 'w') as json_file:
            json.dump(pathDict, json_file, indent=4, ensure_ascii=False)




    def test_loadModel(self):
        # todo yq_register_1101.py
        GPU_iden = 1
        GPU_num = torch.cuda.device_count()
        print('Number of GPU: ' + str(GPU_num))
        for GPU_idx in range(GPU_num):
            GPU_name = torch.cuda.get_device_name(GPU_idx)
            print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
        torch.cuda.set_device(GPU_iden)
        GPU_avai = torch.cuda.is_available()
        print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
        print('If the GPU is available? ' + str(GPU_avai))
        # main()
    def test_readNPZ(self,root,col,row):

        # root = r"Z:\users\yq\MorphDatasets\model\TransMorph\0813\test_result"
        # 形变场数据大小
        channels, depth, height, width = 3, 64, 256, 256
        global_wh = 256 * 4 - 50 * 3
        overlap = 50

        # 假设 deformation_fields 是 4x4 的局部形变场矩阵
        deformation_fields = np.zeros((col, row, channels, depth, height, width))  # 示例数据
        stdy_idx = 0
        for i in range(col):
            for j in range(row):
                # i ,j reverse
                flow = np.load(os.path.join(root, ('disp_{:04d}' + '.npz').format(stdy_idx)))
                flow = flow['arr_0']
                stdy_idx += 1
                deformation_fields[i,j,...] = flow
                # deformation_fields[j, i, ...] = flow

        return deformation_fields
    def test_linerDF2(self):
        import torch
        import numpy as np
        saveRoot = r"Z:\users\yq\MorphDatasets\model\TransMorph\1101\TestResult"

        # 假设 deformation_fields 是 4x4 的局部形变场矩阵，每个形变场大小为 (3, 64, 256, 256)
        # deformation_fields = torch.tensor(np.random.rand(4, 4, 3, 64, 256, 256), dtype=torch.float32)  # 示例数据

        # 重叠区域大小
        overlap_size = 50
        col = 9
        row = 9

        deformation_fields = torch.tensor(self.test_readNPZ(saveRoot,col,row))


        # 加权平均平滑函数
        def weighted_average_smoothing(deformation_fields, overlap_size):
            for i in range(col):
                for j in range(row):
                    if j < 3:  # 水平相邻小图
                        field1 = deformation_fields[i, j]
                        field2 = deformation_fields[i, j + 1]

                        # 提取重叠区域
                        # overlap_field1 = field1[:, :, :, -overlap_size:]  # 图1的右边50像素
                        # overlap_field2 = field2[:, :, :, :overlap_size]  # 图2的左边50像素
                        overlap_field1 = field1[:, :, -overlap_size:, :]  # 图1的下边50像素
                        overlap_field2 = field2[:, :, :overlap_size, :]  # 图2的上边50像素

                        # 生成线性权重
                        weights = torch.linspace(0, 1, overlap_size)
                        # weight1 = weights.view(1, 1, 1, -1)
                        weight1 = weights.view(1, 1, -1, 1)
                        weight2 = 1 - weight1

                        # 加权平均
                        smoothed_overlap = weight1 * overlap_field1 + weight2 * overlap_field2

                        # 更新原始形变场
                        # deformation_fields[i, j, :, :, :, -overlap_size:] = smoothed_overlap
                        # deformation_fields[i, j + 1, :, :, :, :overlap_size] = smoothed_overlap

                        deformation_fields[i, j, :, :, -overlap_size:, :] = smoothed_overlap
                        deformation_fields[i, j + 1, :, :, :overlap_size, :] = smoothed_overlap
                    if i < 3:  # 垂直相邻小图
                        field1 = deformation_fields[i, j]
                        field2 = deformation_fields[i + 1, j]

                        # 提取重叠区域
                        # overlap_field1 = field1[:, :, -overlap_size:, :]  # 图1的下边50像素
                        # overlap_field2 = field2[:, :, :overlap_size, :]  # 图2的上边50像素

                        overlap_field1 = field1[:, :, :, -overlap_size:]  # 图1的右边50像素
                        overlap_field2 = field2[:, :, :, :overlap_size]  # 图2的左边50像素
                        # 生成线性权重
                        weights = torch.linspace(0, 1, overlap_size)
                        # weight1 = weights.view(1, 1, -1, 1)
                        weight1 = weights.view(1, 1, 1, -1)
                        weight2 = 1 - weight1

                        # 加权平均
                        smoothed_overlap = weight1 * overlap_field1 + weight2 * overlap_field2

                        # 更新原始形变场
                        # deformation_fields[i, j, :, :, -overlap_size:, :] = smoothed_overlap
                        # deformation_fields[i + 1, j, :, :, :overlap_size, :] = smoothed_overlap
                        deformation_fields[i, j, :, :, :, -overlap_size:] = smoothed_overlap
                        deformation_fields[i + 1, j, :, :, :, :overlap_size] = smoothed_overlap

            return deformation_fields

        # 应用加权平均平滑方法
        smoothed_fields = weighted_average_smoothing(deformation_fields, overlap_size)

        # # 输出或保存优化后的形变场
        smoothed_fields_np = smoothed_fields.detach().numpy()
        for i in range(col):
            for j in range(row):
                np.save(os.path.join(saveRoot, f'optimized_field_{i}_{j}.npy'), smoothed_fields_np[i, j])
        print("Optimized deformation fields saved.")


    def test_useDF(self):
        root = r"Z:\users\yq\MorphDatasets\model\TransMorph\1101\TestResult"
        stdy_idx = 0
        col = 9
        row = 9
        for i in range(col):
            for j in range(row):
                # flow_path = os.path.join(root, ('disp_{:04d}' + '.npz').format(stdy_idx))
                flow_path = os.path.join(root, f'optimized_field_{i}_{j}.npy')
                x_path = os.path.join(root,"fixed" ,('{:04d}.tif').format(stdy_idx))
                save_path = os.path.join(root, ('refine_out_{:04d}' + '.tif').format(stdy_idx))
                stdy_idx += 1
                useDF(x_path, flow_path, save_path)
                print(f"{i} {j} finished")
                pass

    def test_combineFun(self,imgFormat,savePath, col, row, size):
        # os.path.join(root, ('refine_out_{:04d}' + '.tif'))
        # os.path.join(root,"refine_outall.mha")
        stdy_idx = 0
        overlap_size = 50
        imgAll = np.zeros(size,dtype=np.float32)
        for i in range(col):
            for j in range(row):
                # 计算每个小图在完整图像中的位置
                start_x = i * (256 - overlap_size)
                start_y = j * (256 - overlap_size)
                end_x = start_x + 256
                end_y = start_y + 256

                img_path = imgFormat.format(stdy_idx)
                stdy_idx += 1

                x = sitk.ReadImage(img_path)
                x = sitk.GetArrayFromImage(x)
                # 将当前小图的形变场添加到完整图像的对应位置
                imgAll[:,start_y:end_y , start_x:end_x] = x
                print(f"{i} {j} finished")

        sitk.WriteImage(sitk.GetImageFromArray(imgAll),savePath)

    def test_combineImg(self):
        root = r"Z:\users\yq\MorphDatasets\model\TransMorph\0813\test_result"
        # imgFormat, savePath
        # imgFormat = os.path.join(root, ('y_{:04d}' + '.tif'))
        # savePath = os.path.join(root,"y_out.mha")
        # self.test_combineFun(imgFormat, savePath)

        imgFormat = os.path.join(root, ('x_def_{:04d}' + '.tif'))
        savePath = os.path.join(root, "x_def_out.mha")

        root = r"Z:\users\yq\MorphDatasets\model\TransMorph\1101\TestResult"

        imgFormat = os.path.join(root, ('refine_out_{:04d}' + '.tif'))
        savePath = os.path.join(root, "refine_out.mha")
        imgFormat = root + "/fixed/" + '{:04d}' + '.tif'
        savePath = os.path.join(root, "fixedAllImage.mha")
        size = (64,2000,2000)
        col = 9
        row = 9

        self.test_combineFun(imgFormat, savePath,col, row, size)
