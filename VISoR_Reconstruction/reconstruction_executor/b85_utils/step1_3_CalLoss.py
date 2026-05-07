"""B85 refinement step 1.3 helpers.

This module evaluates block alignment quality and converts block loss files
into refine parameter arrays.
"""

from .common0313 import *
import unittest
import warnings

import numpy as np

from .torch_losses import NCC, NCC_CPU, GlobalNCC, SSIMLoss
from .ome_tiff import write_ome_tiff
import torch, os, time
import SimpleITK as sitk
import multiprocessing

STEP1_3_BLOCK_STATUS_NAME = "step1_3_block_status.txt"
_STEP1_3_BLOCK_STATUS_HEADER = (
    "row\tcol\tstatus\tloss_name\tmoved_name\tfixed_save_name\tmoving_save_name\t"
    "ncc\tssim\treason\tmessage\tupdated_at\n"
)


def _safe_status_field(value):
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _file_ready(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _status_file_path(status_folder):
    return os.path.join(status_folder, STEP1_3_BLOCK_STATUS_NAME)


def _path_from_format(path_format, row_index, col_index):
    return path_format.format(row_index, col_index)


def _loss_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _read_block_statuses(status_path):
    records = {}
    if not os.path.isfile(status_path):
        return records
    with open(status_path, "r", encoding="utf-8-sig", errors="replace") as file:
        for line in file:
            line = line.rstrip("\r\n")
            if not line or line.startswith("row\tcol\tstatus"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                row_index = int(parts[0])
                col_index = int(parts[1])
            except ValueError:
                continue
            records[(row_index, col_index)] = {
                "status": parts[2],
                "loss_name": parts[3] if len(parts) > 3 else "",
                "moved_name": parts[4] if len(parts) > 4 else "",
                "fixed_save_name": parts[5] if len(parts) > 5 else "",
                "moving_save_name": parts[6] if len(parts) > 6 else "",
                "ncc": parts[7] if len(parts) > 7 else "",
                "ssim": parts[8] if len(parts) > 8 else "",
                "reason": parts[9] if len(parts) > 9 else "",
                "message": parts[10] if len(parts) > 10 else "",
                "updated_at": parts[11] if len(parts) > 11 else "",
            }
    return records


def _append_block_status(
        status_path, row_index, col_index, status, loss_name="", moved_name="",
        fixed_save_name="", moving_save_name="", ncc="", ssim="", reason="", message=""):
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    need_header = not os.path.exists(status_path) or os.path.getsize(status_path) == 0
    updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
        row_index,
        col_index,
        _safe_status_field(status),
        _safe_status_field(loss_name),
        _safe_status_field(moved_name),
        _safe_status_field(fixed_save_name),
        _safe_status_field(moving_save_name),
        _safe_status_field(ncc),
        _safe_status_field(ssim),
        _safe_status_field(reason),
        _safe_status_field(message),
        updated_at,
    )
    with open(status_path, "a", encoding="utf-8") as file:
        if need_header:
            file.write(_STEP1_3_BLOCK_STATUS_HEADER)
        file.write(line)
        file.flush()


def _loss_file_ready(path):
    if not _file_ready(path):
        return False
    try:
        read_loss(path)
        return True
    except Exception:
        return False


def _step1_3_outputs_ready(row_index, col_index, moved_format, fixed_save_format,
                           moving_save_format, save_loss_format, save_flag=True):
    loss_path = _path_from_format(save_loss_format, row_index, col_index)
    if not _loss_file_ready(loss_path):
        return False
    moved_path = _path_from_format(moved_format, row_index, col_index)
    if not _file_ready(moved_path):
        return False
    if save_flag:
        if not _file_ready(_path_from_format(fixed_save_format, row_index, col_index)):
            return False
        if not _file_ready(_path_from_format(moving_save_format, row_index, col_index)):
            return False
    return True


def _block_record_complete(status_folder, row_index, col_index, moved_format, fixed_save_format,
                           moving_save_format, save_loss_format, save_flag=True):
    records = _read_block_statuses(_status_file_path(status_folder))
    record = records.get((row_index, col_index))
    if not record:
        return False
    status = record.get("status")
    if status == "evaluated":
        return _step1_3_outputs_ready(
            row_index,
            col_index,
            moved_format,
            fixed_save_format,
            moving_save_format,
            save_loss_format,
            save_flag,
        )
    if status == "failed":
        return True
    return False


def step1_3_block_complete(status_folder, row_index, col_index, moved_format, fixed_save_format,
                           moving_save_format, save_loss_format, save_flag=True):
    return _block_record_complete(
        status_folder,
        row_index,
        col_index,
        moved_format,
        fixed_save_format,
        moving_save_format,
        save_loss_format,
        save_flag,
    )


def bootstrap_step1_3_status(status_folder, offsets, moved_format, fixed_save_format,
                             moving_save_format, save_loss_format, save_flag=True):
    os.makedirs(status_folder, exist_ok=True)
    status_path = _status_file_path(status_folder)
    records = _read_block_statuses(status_path)
    changed = False
    for row_index, col_index in sorted(offsets):
        record = records.get((row_index, col_index))
        if record and _block_record_complete(
                status_folder, row_index, col_index, moved_format, fixed_save_format,
                moving_save_format, save_loss_format, save_flag):
            continue
        if _step1_3_outputs_ready(
                row_index, col_index, moved_format, fixed_save_format,
                moving_save_format, save_loss_format, save_flag):
            loss_path = _path_from_format(save_loss_format, row_index, col_index)
            ncc, ssim = read_loss(loss_path)
            _append_block_status(
                status_path,
                row_index,
                col_index,
                "evaluated",
                os.path.basename(loss_path),
                os.path.basename(_path_from_format(moved_format, row_index, col_index)),
                os.path.basename(_path_from_format(fixed_save_format, row_index, col_index)),
                os.path.basename(_path_from_format(moving_save_format, row_index, col_index)),
                ncc,
                ssim,
                "legacy_existing_outputs",
            )
            changed = True
    return changed


def step1_3_pair_complete(status_folder, offsets, moved_format, fixed_save_format,
                          moving_save_format, save_loss_format, save_flag=True):
    for row_index, col_index in offsets:
        if not _block_record_complete(
                status_folder, row_index, col_index, moved_format, fixed_save_format,
                moving_save_format, save_loss_format, save_flag):
            return False
    return True


def _crop_to_common_size(img1, img2):
    size1 = img1.GetSize()
    size2 = img2.GetSize()
    common_size = [min(size1[axis], size2[axis]) for axis in range(3)]
    if any(size <= 0 for size in common_size):
        raise RuntimeError("No overlapping voxels after size normalization: {} vs {}".format(size1, size2))
    if list(size1) != common_size:
        img1 = img1[:common_size[0], :common_size[1], :common_size[2]]
    if list(size2) != common_size:
        img2 = img2[:common_size[0], :common_size[1], :common_size[2]]
    return img1, img2


def _crop_by_offset(img1, img2, offset, rate):
    off_x = int(offset[0] // rate)
    off_y = int(offset[1] // rate)
    off_z = int(offset[2] // rate)
    max_x, max_y, max_z = img2.GetSize()
    start_x = max(0, -off_x)
    end_x = max_x - max(0, off_x)
    start_y = max(0, -off_y)
    end_y = max_y - max(0, off_y)
    start_z = max(0, -off_z)
    end_z = max_z - max(0, off_z)
    if start_x >= end_x or start_y >= end_y or start_z >= end_z:
        raise RuntimeError(
            "No valid crop after offset. offset={} rate={} size={}".format(offset, rate, img2.GetSize())
        )
    return (
        img1[start_x:end_x, start_y:end_y, start_z:end_z],
        img2[start_x:end_x, start_y:end_y, start_z:end_z],
    )


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
    if not list_:
        return new_array
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
    return array


import multiprocessing
import time, gc


def run_multiprocess(numsThread, taskParas):
    # todo use multiprocess
    if not taskParas:
        print('No step1_3 block tasks to run--')
        return
    if numsThread <= 1:
        for task in taskParas:
            cal_single(*task)
        print('All end--')
        return

    pool = multiprocessing.Pool(numsThread)
    result = []
    for i in range(len(taskParas)):
        msg = 'hello %s' % i
        result.append(pool.apply_async(func=cal_single, args=taskParas[i]))

    pool.close()
    pool.join()

    for res in result:
        res.get()

    print('All end--')


def cal_single(key, value, rate, spacing
               , movingFormat, movedFormat, FixedPathFormat,
               MovedPathFormat, FixedsaveRefineFormat,
               MovingsaveRefineFormat, save_loss_format, saveFlag,
               status_folder=None, force=False):
    i = key[0];
    j = key[1];
    if status_folder is None:
        status_folder = os.path.dirname(save_loss_format.format(i, j))
    status_path = _status_file_path(status_folder)
    if (
            not force
            and step1_3_block_complete(
                status_folder,
                i,
                j,
                movedFormat,
                FixedsaveRefineFormat,
                MovingsaveRefineFormat,
                save_loss_format,
                saveFlag,
            )
    ):
        print(f"Step1_3 block {i}_{j} already complete from status file: {status_path}")
        return

    try:
        ncc_loss = GlobalNCC().cuda()  # NCCCUDA?
        ssim_loss = SSIMLoss(spatial_dims=3).cuda()  # NCCCUDA?
        moving = sitk.ReadImage(movingFormat.format(i, j))[:, :, :-10]
        movedPath = (movedFormat.format(i, j))
        moving.SetOrigin([0, 0, 0])
        moving.SetSpacing(spacing)

        translate = value
        translation = sitk.TranslationTransform(3, translate)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(moving)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(translation)
        resampled_image = resampler.Execute(moving)
        write_ome_tiff(resampled_image, movedPath)

        img1 = (sitk.ReadImage(FixedPathFormat.format(i, j)))[:, :, 10:]
        img2 = (sitk.ReadImage(MovedPathFormat.format(i, j)))
        img1.SetOrigin([0, 0, 0])
        img2.SetOrigin([0, 0, 0])
        img1, img2 = _crop_to_common_size(img1, img2)
        img1, img2 = _crop_by_offset(img1, img2, value, rate)

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

        tensor_img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor_img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        shape = img1.shape
        if shape[0] > 7:
            loss = ncc_loss(tensor_img1, tensor_img2)
            data_range = max(tensor_img1.max(), tensor_img2.max())
            ssim_res = ssim_loss(tensor_img1, tensor_img2, data_range)
        else:
            loss = 0
            ssim_res = 0
        loss_value = _loss_value(loss)
        ssim_value = _loss_value(ssim_res)
        save_loss_path = save_loss_format.format(i, j)
        with open(save_loss_path, "w") as f:
            f.write(f"ncc:{loss_value},ssim:{ssim_value}")
        _append_block_status(
            status_path,
            i,
            j,
            "evaluated",
            os.path.basename(save_loss_path),
            os.path.basename(movedPath),
            os.path.basename(FixedsaveRefineFormat.format(i, j)),
            os.path.basename(MovingsaveRefineFormat.format(i, j)),
            loss_value,
            ssim_value,
            "written",
        )
        print(f"key {key} is NCC loss:{loss_value}; SSIM loss:{ssim_value}; value is {value}")
    except Exception as exc:
        _append_block_status(
            status_path,
            i,
            j,
            "failed",
            "",
            "",
            "",
            "",
            "",
            "",
            exc.__class__.__name__,
            str(exc),
        )
        print("Step1_3 block {}_{} failed: {}".format(i, j, exc))


def CalNCC(spacing, movingFormat, movedFormat,
           Offsets, FixedPathFormat, MovedPathFormat,
           FixedsaveRefineFormat, MovingsaveRefineFormat,
           save_loss_format,
           rate,
           status_folder=None,
           force=False,
           num_threads=20):
    # img1img2?D [depth, height, width]
    # 

    saveFlag = True;
    # FixedsaveRefineFormat = r"Z:\users\yq\MorphDatasets\TestTemp\th2_refine\{}_{}up_temp_all.tif"
    # MovingsaveRefineFormat = r"Z:\users\yq\MorphDatasets\TestTemp\th2_refine\{}_{}moved.tif"
    lossList = {}
    ssimList = {}
    task_chunks = []
    if status_folder is None:
        status_folder = os.path.dirname(save_loss_format.format(0, 0))
    bootstrap_step1_3_status(
        status_folder,
        Offsets,
        movedFormat,
        FixedsaveRefineFormat,
        MovingsaveRefineFormat,
        save_loss_format,
        saveFlag,
    )
    for key, value in Offsets.items():
        if (
                not force
                and step1_3_block_complete(
                    status_folder,
                    key[0],
                    key[1],
                    movedFormat,
                    FixedsaveRefineFormat,
                    MovingsaveRefineFormat,
                    save_loss_format,
                    saveFlag,
                )
        ):
            continue
        temp = (key, value, rate, spacing
                , movingFormat, movedFormat, FixedPathFormat,
                MovedPathFormat, FixedsaveRefineFormat,
                MovingsaveRefineFormat, save_loss_format, saveFlag,
                status_folder, force)
        task_chunks.append(temp)
        # cal_single(key,value,rate,spacing
        #        ,movingFormat,movedFormat, FixedPathFormat,
        #        MovedPathFormat,FixedsaveRefineFormat,
        #        MovingsaveRefineFormat,save_loss_format,saveFlag)

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
    if len(filtered_arr) == 0:
        filtered_arr = arr_np
        dropped_arr = np.array([])
    if len(filtered_arr) > 0:
        filtered_mean = np.mean(filtered_arr)
    else:
        filtered_mean = 0
    print(
        "Step1_3 offset filter: total={} kept={} dropped={} mean={:.2f}".format(
            len(arr),
            len(filtered_arr),
            len(dropped_arr),
            filtered_mean,
        )
    )
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

