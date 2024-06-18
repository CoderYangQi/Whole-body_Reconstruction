'''
@ yangqi
Translate_Npy 应用paras 去配准

'''


import unittest

import numpy as np

from common_script.Torch_Losses import NCC_CPU
import torch, os
import SimpleITK as sitk
def ReadOffsetTxt(txtPath = r"Z:\users\yq\MorphDatasets\TestTemp\th2_33\tf_33_pars.txt"):
    import re

    # 初始化一个空字典来存储坐标和对应的偏移量
    offsets = {}

    # 假设文本文件的内容已经以字符串形式给出或已经保存在一个文件中

    with open(txtPath, 'r') as file:
        data = file.readlines()

    file.close()
    # 按行分割数据
    # lines = data.split('\n')

    # 处理每一行
    for line in data:
        # 使用正则表达式提取坐标和偏移量
        match = re.match(r'\[(\d+), (\d+)\]: \(([^)]+)\)', line)
        if match:
            # 提取坐标和偏移量
            coord = (int(match.group(1)), int(match.group(2)))
            offsets_values = match.group(3).split(',')
            offsets[coord] = tuple(float(v) for v in offsets_values)

    # 打印结果
    tempName = 'th2_33'
    spacing = [4.0, 4.0, 4.0]
    for key, value in offsets.items():
        print(f"key is {key}; value is {value}")
        i = key[0];
        j = key[1];
    return offsets

def Translate_Npy(offsets,movingFormat, movedFormat):
    # 打印结果
    # spacing = [4.0, 4.0, 4.0]
    spacing = [1.0, 1.0, 1.0]

    # moving = sitk.ReadImage(os.path.join(r"D:\USERS\yq\code\cal_overlap\Refine", tempName,
    #                                      str(i) + "_" + str(j) + "down_temp_all.tif"))

    for key, value in offsets.items():
        print(f"key is {key}; value is {value}")
        i = key[0];
        j = key[1];
        moving = sitk.ReadImage(movingFormat.format(i,j))
        movedPath = (movedFormat.format(i,j))
        moving.SetOrigin([0,0,0])
        moving.SetSpacing(spacing)

        # todo translate
        # 平移向量
        translate = value

        # 创建平移变换
        translation = sitk.TranslationTransform(3, translate)
        # 重采样过滤器
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(moving)  # 设置参考图像
        resampler.SetInterpolator(sitk.sitkLinear)  # 设置插值方法
        resampler.SetTransform(translation)  # 设置应用的平移变换

        # 用平移变换重采样图像
        resampled_image = resampler.Execute(moving)

        # 保存重采样后的图像
        sitk.WriteImage(resampled_image, movedPath)
def linerArray(array):
    # 找到所有0值的位置
    zero_positions = np.where(array == 0)
    # print(zero_positions)

    # 对每个0值进行插值处理
    for pos in zip(zero_positions[0], zero_positions[1]):
        # 提取周围的元素
        neighbors = []
        # 检查上下左右四个方向
        for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n_row, n_col = pos[0] + d[0], pos[1] + d[1]
            # 确保索引在数组范围内
            if 0 <= n_row < array.shape[0] and 0 <= n_col < array.shape[1]:
                neighbors.append(array[n_row, n_col])

        # 计算非零邻居的平均值，只考虑非零值
        if neighbors:
            non_zero_neighbors = [n for n in neighbors if n != 0]
            if non_zero_neighbors:
                array[pos] = sum(non_zero_neighbors) / len(non_zero_neighbors)
            else:
                # 如果周围没有非零值，可以选择一个默认值，例如周围数的平均或其他合理的值
                array[pos] = np.mean(array)

    # 打印结果
    print(array)
    return  array
def CalNCC(Offsets,FixedPathFormat, MovedPathFormat, FixedsaveRefineFormat,MovingsaveRefineFormat,rate = 1):
    # 假设img1和img2是你的3D图像数组，形状为 [depth, height, width]
    # 例如，使用随机数据来创建这些图像

    saveFlag = True;
    # FixedsaveRefineFormat = r"Z:\users\yq\MorphDatasets\TestTemp\th2_refine\{}_{}up_temp_all.tif"
    # MovingsaveRefineFormat = r"Z:\users\yq\MorphDatasets\TestTemp\th2_refine\{}_{}moved.tif"
    lossList = {}
    for key, value in Offsets.items():
        # test i = 7; j = 3; todo 选取出有效的已对齐的区域
        # i = 7; j = 3;
        i = key[0];
        j = key[1]
        temp = value
        # off_ = (i,j)
        # temp = Offsets[off_]
        off_z = temp[2] // rate;
        off_y = temp[1] // rate;
        off_x = temp[0] // rate
        # img1 = (sitk.ReadImage(r"Z:\users\yq\MorphDatasets\TestTemp\th2_33\{}_{}up_temp_all.tif".format(i, j)))
        # img2 = (sitk.ReadImage(r"Z:\users\yq\MorphDatasets\TestTemp\th2_33\{}_{}moved.tif".format(i, j)))
        img1 = (sitk.ReadImage(FixedPathFormat.format(i, j)))
        img2 = (sitk.ReadImage(MovedPathFormat.format(i, j)))
        max_z = img2.GetSize()[2]
        max_y = img2.GetSize()[1]
        max_x = img2.GetSize()[0]
        if off_z < 0:
            start = int(- off_z);
            end = max_z
        else:
            start = 0;
            end = int(max_z - off_z)

        # y axis
        if off_y < 0:
            start_y = int(-off_y);
            end_y = max_y
        else:
            start_y = 0;
            end_y = int(max_x - off_y)

        # x axis
        if off_x < 0:
            start_x = int(-off_x);
            end_x = max_y
        else:
            start_x = 0;
            end_x = int(max_x - off_x)

        img1 = img1[start_x:end_x, start_y:end_y, start:end]
        img2 = img2[start_x:end_x, start_y:end_y, start:end]

        if saveFlag:
            sitk.WriteImage(img1, FixedsaveRefineFormat.format(i, j))
            sitk.WriteImage(img2, MovingsaveRefineFormat.format(i, j))

        img1 = sitk.GetArrayFromImage(img1)
        img2 = sitk.GetArrayFromImage(img2)
        # todo 计算 loss
        # # img1 = np.random.rand(64, 64, 64)
        # # img2 = np.random.rand(64, 64, 64)
        # gpu_id = '0'
        # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        # # device = torch.device("cuda:1")
        # # torch.cuda.set_device(eval(gpu_id))  # 单卡
        # # # 将NumPy数组转换为PyTorch Tensor
        # tensor_img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        # tensor_img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        # # tensor_img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        # # tensor_img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        # # 将NumPy数组转换为PyTorch Tensor
        # # tensor_img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # # tensor_img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # # 初始化NCC模块
        # ncc_loss = NCC().cuda()  # 确保NCC模块在CUDA上运行
        # # ncc_loss = NCC()  # 确保NCC模块在CUDA上运行

        tensor_img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor_img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        ncc_loss = NCC_CPU()  # 确保NCC模块在CUDA上运行

        # 计算两个图像之间的NCC误差
        loss = ncc_loss(tensor_img1, tensor_img2)
        lossList[key] = loss
        # selfloss = ncc_loss(tensor_img1, tensor_img1)
        print(f"key {key} is NCC loss:{loss.item()}; value is {value}")
    return lossList
def readCSV(csv_path):
    # Path to the uploaded CSV file
    pointList = []
    if os.path.exists(csv_path):
        # 打开文件并读取每一行
        with open(csv_path, 'r') as file:
            next(file)
            for line in file:
                # 使用 strip() 移除行末的换行符，然后用 split(',') 分割每行
                data = line.strip().split(',')
                pointList.append([int(data[5]),int(data[6]),int(data[8])])
                print(data)
    print(pointList)
    return pointList


class Calloss(unittest.TestCase):
    def test_print(self):
        print("start")
    def test_ReadNpy(self):

        # 初始化一个空字典来存储坐标和对应的偏移量
        offsets = {}




    def test_loss(self):
        # root = r"D:\USERS\yq\code\cal_overlap\Refine\th2_0528"
        # root = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\th2_0528"
        root = r"D:\USERS\yq\TH2_Reconstruction\ROI_130_151\th2_0602"
        # csvPath = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\surface\points\84_250_300.csv"
        # pointList = readCSV(csvPath)
        # size = [8000, 7200]
        size = [11200, 9200]
        block_size = 500
        # todo
        # Calculate row_index, col_index for each position
        transformed_data = []
        # transformed_data = [[pos[0] // block_size, pos[1] // block_size, pos[2] - 300] for pos in pointList]
        for slice_index in range(130,133):
            rate = 2
            Offsets = ReadOffsetTxt(os.path.join(root,"tf_{}_pars.txt".format(slice_index)))

            refinePath = os.path.join(root,fr"{slice_index}_{slice_index+1}")
            FixedsaveRefineFormat = os.path.join(refinePath, "save_{}_{}up_temp_all.tif")
            MovingsaveRefineFormat = os.path.join(refinePath, "save_{}_{}moved.tif")
            if not os.path.exists(refinePath):
                os.mkdir(refinePath)
            # {}_{}moved.tif

            fixedFormat = os.path.join(root, f"{slice_index}_{slice_index + 1}",
                                        "{}_{}" + "up_temp_all.tif")
            movingFormat = os.path.join(root, f"{slice_index}_{slice_index+1}",
                                                              "{}_{}" + "down_temp_all.tif")
            movedFormat = os.path.join(root, f"{slice_index}_{slice_index+1}",
                         "{}_{}" + "moved.tif")
            Translate_Npy(Offsets,movingFormat,movedFormat)
            lossList = CalNCC(Offsets,fixedFormat, movedFormat, FixedsaveRefineFormat,MovingsaveRefineFormat)

            # npy_path = os.path.join(root,"tf_{}_pars.npy")
            # npy_array = np.load(npy_path.format(slice_index))

            row = int(np.floor(size[0] / block_size))
            col = int(np.floor(size[1] / block_size))
            vector_points = np.zeros((row, col, 3))
            npy_array = vector_points
            print(f"shape of npy array is {size}")
            # 一般来说选取数据的均值 mean 作为浮动的基准面
            sum = 0
            ct = 0
            for key, value in Offsets.items():
                # if value[2] < -50:
                i = key[0];
                j = key[1]
                if lossList[key] < -0.20:
                    ct += 1
                    print(f"key is {key}; value is {value}; loss is {lossList[key]} ct is {ct}")


                    sum += value[2]
                    npy_array[i, j] = np.array([value[0] * rate, value[1] * rate, value[2]])
                    continue
                npy_array[i, j] = np.array([0, 0, 0])
            if ct == 0:
                print(f"{slice_index} is failed")
                continue

            # todo
            if not transformed_data:
                for point in transformed_data:
                    i = point[0];
                    j = point[1]
                    npy_array[i, j, 2] = point[2]
            non_zero_count = np.count_nonzero(npy_array[:,:,2])
            sum = np.sum(npy_array[:,:,2])
            mean_sum = sum / non_zero_count
            print(f"mean sum is {mean_sum}")
            z_array = npy_array[:, :, 2]
            # 将0填充为标准面 然后再减去 然后再插值 对z轴进行操作
            non_zero_positions = np.where(z_array != 0)
            for pos in zip(non_zero_positions[0], non_zero_positions[1]):
                z_array[pos[0], pos[1]] = z_array[pos[0], pos[1]] - mean_sum
            # print(non_zero_positions)
            temp_z = abs(z_array)
            index = np.where(temp_z > 50)
            z_array[index] = 0
            print(z_array)
            unuseful_pos = np.where(z_array == 0)
            useful_pos = np.where(z_array != 0)
            for pos in zip(unuseful_pos[0], unuseful_pos[1]):
                npy_array[pos[0], pos[1],0:2] = 0
            x_array = npy_array[:, :, 0]
            y_array = npy_array[:, :, 1]
            # 根据 z 的信息对 x y 轴都进行操作
            x_array[index] = 0
            y_array[index] = 0

            npy_size = npy_array.shape
            total_size = npy_size[0] * npy_size[1]
            mean_x = np.mean(x_array) * total_size / (total_size - unuseful_pos[0].size)
            mean_y = np.mean(x_array) * total_size / (total_size - unuseful_pos[0].size)
            for pos in zip(useful_pos[0], useful_pos[1]):
                x_array[pos[0], pos[1]] = x_array[pos[0], pos[1]] - mean_x
                y_array[pos[0], pos[1]] = y_array[pos[0], pos[1]] - mean_y

            z = linerArray(z_array)
            z += mean_sum
            y =  linerArray(y_array)
            x =  linerArray(x_array)
            x += mean_x
            y += mean_y
            result = np.zeros(npy_size)
            result[:,:,0] = x; result[:,:,1] = y; result[:,:,2] = z;
            np.save(os.path.join(root,"0526_refine_{}_pars.npy".format(slice_index)),result)
