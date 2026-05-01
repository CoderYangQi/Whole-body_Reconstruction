"""B85 refinement step 1.3 helpers.

This module evaluates block alignment quality and converts block loss files
into refine parameter arrays.
"""

from .common0313 import *
import unittest
import warnings

import numpy as np

from .torch_losses import NCC, NCC_CPU, GlobalNCC, SSIMLoss
import torch, os, time
import SimpleITK as sitk
import multiprocessing


def ReadOffsetTxt(txtPath=r"Z:\users\yq\MorphDatasets\TestTemp\th2_33\tf_33_pars.txt"):
    import re

    # 
    offsets = {}

    # 

    with open(txtPath, 'r') as file:
        data = file.readlines()

    file.close()
    # 
    # lines = data.split('\n')

    # ?
    for line in data:
        # ?
        match = re.match(r'\[(\d+), (\d+)\]: \(([^)]+)\)', line)
        if match:
            # 
            coord = (int(match.group(1)), int(match.group(2)))
            offsets_values = match.group(3).split(',')
            offsets[coord] = tuple(float(v) for v in offsets_values)

    # 
    tempName = 'th2_33'
    spacing = [4.0, 4.0, 4.0]
    for key, value in offsets.items():
        print(f"key is {key}; value is {value}")
        i = key[0];
        j = key[1];
    return offsets


def Translate_Npy(offsets, movingFormat, movedFormat, spacing):
    import re
    # 

    # spacing = [1.0, 1.0, 1.0]

    # moving = sitk.ReadImage(os.path.join(r"D:\USERS\yq\code\cal_overlap\Refine", tempName,
    #                                      str(i) + "_" + str(j) + "down_temp_all.tif"))

    for key, value in offsets.items():
        print(f"key is {key}; value is {value}")
        i = key[0];
        j = key[1];
        moving = sitk.ReadImage(movingFormat.format(i, j))
        movedPath = (movedFormat.format(i, j))
        moving.SetOrigin([0, 0, 0])
        moving.SetSpacing(spacing)

        # todo translate
        # 
        translate = value

        # 
        translation = sitk.TranslationTransform(3, translate)
        # 
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(moving)  # ?
        resampler.SetInterpolator(sitk.sitkLinear)  # ?
        resampler.SetTransform(translation)  # ?

        # 
        resampled_image = resampler.Execute(moving)

        # ?
        write_ome_tiff(resampled_image, movedPath)


def linerArray2D(array, zero_positions, non_zero_positions):
    list_ = []
    new_array = np.zeros(array.shape)
    new_array = array.copy()
    for pos in zip(non_zero_positions[0], non_zero_positions[1]):
        list_.append(array[pos[0], pos[1]])
    mean = np.mean(list_)

    # ??
    for pos in zip(zero_positions[0], zero_positions[1]):
        # ?
        neighbors = []
        # ?
        for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n_row, n_col = pos[0] + d[0], pos[1] + d[1]
            # 
            if 0 <= n_row < array.shape[0] and 0 <= n_col < array.shape[1]:
                neighbors.append(array[n_row, n_col])

        # ?
        if neighbors:
            non_zero_neighbors = [n for n in neighbors if n != 0]
            if non_zero_neighbors:
                new_array[pos] = sum(non_zero_neighbors) / len(non_zero_neighbors)
            else:
                # ?
                # new_array[pos] = np.mean(array)
                new_array[pos] = mean
        # else:
        #     new_array[pos] = mean
    # 
    print(new_array)
    return new_array


def linerArray(array):
    # ?
    zero_positions = np.where(array == 0)
    # print(zero_positions)

    # ??
    for pos in zip(zero_positions[0], zero_positions[1]):
        # ?
        neighbors = []
        # ?
        for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n_row, n_col = pos[0] + d[0], pos[1] + d[1]
            # 
            if 0 <= n_row < array.shape[0] and 0 <= n_col < array.shape[1]:
                neighbors.append(array[n_row, n_col])

        # ?
        if neighbors:
            non_zero_neighbors = [n for n in neighbors if n != 0]
            if non_zero_neighbors:
                array[pos] = sum(non_zero_neighbors) / len(non_zero_neighbors)
            else:
                # ?
                array[pos] = np.mean(array)

    # 
    print(array)
    return array


import multiprocessing
import time, gc


def run_multiprocess(numsThread, taskParas):
    # todo use multiprocess
    pool = multiprocessing.Pool(numsThread)
    result = []
    for i in range(len(taskParas)):
        msg = 'hello %s' % i
        result.append(pool.apply_async(func=cal_single, args=taskParas[i]))

    pool.close()
    pool.join()

    # for res in result:
    #     print('***:', res.get())  # get()?

    print('All end--')


def cal_single(key, value, rate, spacing
               , movingFormat, movedFormat, FixedPathFormat,
               MovedPathFormat, FixedsaveRefineFormat,
               MovingsaveRefineFormat, save_loss_format, saveFlag):
    ncc_loss = GlobalNCC().cuda()  # NCCCUDA?
    ssim_loss = SSIMLoss(spatial_dims=3).cuda()  # NCCCUDA?
    i = key[0];
    j = key[1];
    moving = sitk.ReadImage(movingFormat.format(i, j))[:, :, :-10]
    movedPath = (movedFormat.format(i, j))
    moving.SetOrigin([0, 0, 0])
    moving.SetSpacing(spacing)

    # todo translate
    # 
    translate = value

    # 
    translation = sitk.TranslationTransform(3, translate)
    # 
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(moving)  # ?
    resampler.SetInterpolator(sitk.sitkLinear)  # ?
    resampler.SetTransform(translation)  # ?

    # 
    resampled_image = resampler.Execute(moving)

    # ?
    # print(movedPath)
    write_ome_tiff(resampled_image, movedPath)

    # test i = 7; j = 3; todo 
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
    img1 = (sitk.ReadImage(FixedPathFormat.format(i, j)))[:, :, 10:]
    img2 = (sitk.ReadImage(MovedPathFormat.format(i, j)))
    img1.SetOrigin([0, 0, 0])
    img2.SetOrigin([0, 0, 0])
    max_z = img2.GetSize()[2]
    max_y = img2.GetSize()[1]
    max_x = img2.GetSize()[0]
    if img1.GetSize()[2] != max_z:
        raise "error img1.GetSize()[2] != img2.GetSize()[2]"
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

    if start > end:
        warnings.warn(
            "The 'start' value should not exceed 'end'. Unexpected results may occur.",
            category=UserWarning,  # [1,4](@ref)
            stacklevel=2  # [5,7](@ref)
        )

    # 
    threshold = 120
    pixel_type = img1.GetPixelID()
    if pixel_type == sitk.sitkUInt8:
        print("Image pixel type is sitkUInt8; preprocessing is skipped.")
    elif pixel_type == sitk.sitkUInt16:
        print("Image pixel type is sitkUInt16; preprocessing is applied.")
        img1 = Preprocess(img1, threshold)
    if saveFlag:
        write_ome_tiff(img1, FixedsaveRefineFormat.format(i, j))
        write_ome_tiff(img2, MovingsaveRefineFormat.format(i, j))
    pixel_type = img2.GetPixelID()
    if pixel_type == sitk.sitkUInt8:
        print("Image pixel type is sitkUInt8; preprocessing is skipped.")
    elif pixel_type == sitk.sitkUInt16:
        print("Image pixel type is sitkUInt16; preprocessing is applied.")
        img2 = Preprocess(img2, threshold)
    img1 = sitk.GetArrayFromImage(img1)
    img2 = sitk.GetArrayFromImage(img2)
    # todo  loss

    tensor_img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor_img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    shape = img1.shape
    if shape[0] > 7:
        # NCC

        loss = ncc_loss(tensor_img1, tensor_img2)
        data_range = max(tensor_img1.max(), tensor_img2.max())
        ssim_res = ssim_loss(tensor_img1, tensor_img2, data_range)
        # selfloss = ncc_loss(tensor_img1, tensor_img1)
    else:
        loss = 0
        ssim_res = 0
    # lossList[key] = loss
    # ssimList[key] = ssim_res
    save_loss_path = save_loss_format.format(i, j)
    with open(save_loss_path, "w") as f:
        f.write(f"ncc:{loss},ssim:{ssim_res}")
    print(f"key {key} is NCC loss:{loss}; SSIM loss:{ssim_res}; value is {value}")


def CalNCC(spacing, movingFormat, movedFormat,
           Offsets, FixedPathFormat, MovedPathFormat,
           FixedsaveRefineFormat, MovingsaveRefineFormat,
           save_loss_format,
           rate):
    # img1img2?D [depth, height, width]
    # 

    saveFlag = True;
    # FixedsaveRefineFormat = r"Z:\users\yq\MorphDatasets\TestTemp\th2_refine\{}_{}up_temp_all.tif"
    # MovingsaveRefineFormat = r"Z:\users\yq\MorphDatasets\TestTemp\th2_refine\{}_{}moved.tif"
    lossList = {}
    ssimList = {}
    task_chunks = []
    for key, value in Offsets.items():
        if os.path.exists(save_loss_format.format(key[0], key[1])):
            print(f"exists {save_loss_format.format(key[0], key[1])}")
            continue
        temp = (key, value, rate, spacing
                , movingFormat, movedFormat, FixedPathFormat,
                MovedPathFormat, FixedsaveRefineFormat,
                MovingsaveRefineFormat, save_loss_format, saveFlag)
        task_chunks.append(temp)
        # cal_single(key,value,rate,spacing
        #        ,movingFormat,movedFormat, FixedPathFormat,
        #        MovedPathFormat,FixedsaveRefineFormat,
        #        MovingsaveRefineFormat,save_loss_format,saveFlag)

    num_threads = 20
    run_multiprocess(num_threads, task_chunks)
    return lossList, ssimList


def readCSV(csv_path):
    # Path to the uploaded CSV file
    pointList = []
    if os.path.exists(csv_path):
        # ?
        with open(csv_path, 'r') as file:
            next(file)
            for line in file:
                #  strip()  split(',') 
                data = line.strip().split(',')
                pointList.append([int(data[5]), int(data[6]), int(data[8])])
                print(data)
    print(pointList)
    return pointList


import csv


def read_loss(file_path):
    # file_path = r"K:\STZ1_914#\save_temp_0313\temp_block\160_161\loss_10_12.txt"
    results = {"ncc": [], "ssim": []}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # 
            pairs = line.split(',')
            for pair in pairs:
                key, value = pair.split(':')
                key = key.strip().lower()  # 
                if key in results:
                    results[key].append(float(value))
    # print(results)
    return results["ncc"][0], results["ssim"][0]


def filter_arr(arr):
    #  NumPy 
    arr_np = np.array(arr)

    # ?
    mean = np.mean(arr_np)
    std_dev = np.std(arr_np)

    # ?2 
    lower_bound = mean - 1 * std_dev
    upper_bound = mean + 1 * std_dev

    # ?
    filtered_arr = arr_np[(arr_np >= lower_bound) & (arr_np <= upper_bound)]

    dropped_arr = arr_np[(arr_np < lower_bound) | (arr_np > upper_bound)]

    # ?
    filtered_mean = np.mean(filtered_arr)

    print(f"? {arr}")
    print(f"? {lower_bound:.2f} ?{upper_bound:.2f}")
    print(f"? {filtered_arr.tolist()}")
    print(f"? {filtered_mean:.2f}")
    return filtered_arr, dropped_arr


def read_coordinates(file_path):
    """
     (x, y, z) ?
     [(-31.8817, -44.628, -49.7399), ...]
    """
    coordinates = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    # ?
                    stripped = line.strip('()\n')
                    values = [float(x.strip()) for x in stripped.split(',')]

                    # 
                    if len(values) == 3:
                        coordinates.append(values)
                    else:
                        print(f"? {line}")
            print(coordinates)
            return coordinates

    except FileNotFoundError:
        print(f":  {file_path} ")
    except Exception as e:
        print(f"? {str(e)}")


def rest(used_list, Offsets, slice_index,
         ct, npy_array, rate, transformed_data
         , root):
    filit_list, dropped_list = filter_arr(used_list)
    # sec_filit_list,_ = filter_arr(filit_list)

    left_offset_dict = {}
    for key, value in Offsets.items():
        if value[2] in filit_list:
            left_offset_dict[key] = value

    # if ct == 0:
    #     print(f"{slice_index} is failed")
    #     return
    # copy data
    for key, value in left_offset_dict.items():
        i = key[0];
        j = key[1]
        npy_array[i, j] = np.array([value[0] * rate, value[1] * rate, value[2] * rate])

    # todo
    if not transformed_data:
        for point in transformed_data:
            i = point[0];
            j = point[1]
            npy_array[i, j, 2] = point[2]
    # non_zero_count = np.count_nonzero(npy_array[:, :, 2])
    # sum = np.sum(npy_array[:, :, 2])
    # mean_sum = sum / non_zero_count

    np.save(os.path.join(root, "{}_np_array.npy".format(slice_index)), npy_array)
    npy_array = np.load(os.path.join(root, "{}_np_array.npy".format(slice_index)))

    # print(f"{np.sum(npy_array1 - npy_array)}")
    # ?
    # mean_sum = np.mean(filit_list)
    # print(f"mean sum is {mean_sum}")

    z_array = npy_array[:, :, 2]
    # ? ??z?
    non_zero_positions = np.where(z_array != 0)
    zero_positions = np.where(z_array == 0)
    z = linerArray2D(z_array, zero_positions, non_zero_positions)
    x_array = npy_array[:, :, 0]
    y_array = npy_array[:, :, 1]
    y = linerArray2D(y_array, zero_positions, non_zero_positions)

    x = linerArray2D(x_array, zero_positions, non_zero_positions)

    npy_size = npy_array.shape
    # #  z  x y 
    result = np.zeros(npy_size)
    result[:, :, 0] = x;
    result[:, :, 1] = y;
    result[:, :, 2] = z;
    np.save(os.path.join(root, "refine_{}_pars.npy".format(slice_index)), result)


def cal_loss():
    # root = r"D:\USERS\yq\code\cal_overlap\Refine\th2_0528"
    # root = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\th2_0528"
    root = r"E:\20250426_SMY_TAC1_AI14_1_1\Reconstruction\tac\temp_block"
    txt_root = r"E:\20250426_SMY_TAC1_AI14_1_1\Reconstruction\tac\minus10"

    # csvPath = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\surface\points\84_250_300.csv"
    # pointList = readCSV(csvPath)
    # size = [8000, 7200]
    size = [10500, 6750]

    # max size is [10401  6032]
    block_size = 250
    # ncc_loss = NCC_CPU()  # NCCCUDA?

    # todo
    # Calculate row_index, col_index for each position
    transformed_data = []
    # transformed_data = [[pos[0] // block_size, pos[1] // block_size, pos[2] - 300] for pos in pointList]
    for slice_index in range(36, 42):
        print(f"start {slice_index}")
        res_path = os.path.join(root, "refine_{}_pars.npy".format(slice_index))
        # if os.path.exists(res_path):
        #     print(f"exists {res_path}")
        #     continue
        Offsets = {}
        temp_block_folder = os.path.join(root, f"{slice_index}_{slice_index + 1}")
        txt_block_folder = os.path.join(txt_root, f"{slice_index}_{slice_index + 1}")
        row = int(np.floor(size[0] / block_size))
        col = int(np.floor(size[1] / block_size))
        for i in range(row):
            for j in range(col):
                pos_path = os.path.join(txt_block_folder, f"pos_{i}_{j}.txt")
                print(pos_path)
                if os.path.exists(pos_path):
                    res = read_coordinates(pos_path)
                    # if not res:
                    #     continue
                    coord = res[0]
                    Offsets[(int(i), int(j))] = coord

        rate = 1
        if not Offsets:
            print(f"index {slice_index} is empty")
            continue
        refinePath = os.path.join(root, fr"{slice_index}_{slice_index + 1}")
        FixedsaveRefineFormat = os.path.join(refinePath, "save_{}_{}up_temp_all.tif")
        MovingsaveRefineFormat = os.path.join(refinePath, "save_{}_{}moved.tif")
        if not os.path.exists(refinePath):
            os.mkdir(refinePath)
        # {}_{}moved.tif

        fixedFormat = os.path.join(root, f"{slice_index}_{slice_index + 1}",
                                   "{}_{}" + "up_temp_all.tif")
        movingFormat = os.path.join(root, f"{slice_index}_{slice_index + 1}",
                                    "{}_{}" + "down_temp_all.tif")
        movedFormat = os.path.join(root, f"{slice_index}_{slice_index + 1}",
                                   "{}_{}" + "moved.tif")
        # Translate_Npy(Offsets, movingFormat, movedFormat,spacing = [4.0, 4.0, 4.0])
        spacing = [4.0, 4.0, 4.0]
        FixedPathFormat = fixedFormat
        MovedPathFormat = movedFormat
        save_loss_format = os.path.join(txt_block_folder, "loss_{}_{}.txt")
        # CalNCC(spacing,movingFormat,movedFormat,
        #    Offsets,FixedPathFormat, MovedPathFormat,
        #    FixedsaveRefineFormat,MovingsaveRefineFormat,
        #     save_loss_format,
        #    rate=4)

        loss_dict = {}
        used_offsets = {}
        used_list = []
        ct = 0

        for i in range(row):
            for j in range(col):
                loss_path = os.path.join(txt_block_folder, f"loss_{i}_{j}.txt")
                pos = (i, j)
                if os.path.exists(loss_path):
                    ncc, ssim = read_loss(loss_path)
                    if ncc > 0.80 and ssim > 0.50:
                        ct += 1
                        value = Offsets[pos]
                        used_offsets[pos] = value
                        used_list.append(value[2])  # offset
                        print(f"key is {pos}; value is {value}; ncc: {ncc}; ssim: {ssim} ct is {ct}")
        if len(used_list):
            vector_points = np.zeros((row, col, 3))
            npy_array = vector_points
            rest(used_list, Offsets, slice_index,
                 ct, npy_array, rate, transformed_data
                 , root)

        print()


if __name__ == '__main__':
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    path = None
    cal_loss()
    # read_loss()

