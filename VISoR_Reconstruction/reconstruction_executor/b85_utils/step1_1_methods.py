"""B85 refinement step 1.1 helpers.

This module creates adjacent-slice image blocks used for coarse alignment.
It is a local copy of the B85 utility code so the runner does not depend on
YQReconstructionScripts at runtime.
"""
from .yq_elastix_files import *
import numpy as np
import os
import tifffile

import unittest
from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
from VISoR_Brain.utils.elastix_files import *
from VISoR_Reconstruction.reconstruction.brain_reconstruct_methods.common import fill_outside
from .common0313 import *
from .ome_tiff import write_ome_tiff

def CalSurfaceTranslate(prev_surface_path, next_surface_path):
    def PreProcess(img):
        img = sitk.Cast(img, sitk.sitkFloat32)
        refineImg = sitk.Clamp((sitk.Log(sitk.Cast(img, sitk.sitkFloat32)) - 4.6) * 39.4, sitk.sitkUInt8, 0, 255)
        return refineImg
    translateDict = {}

    #  index ?33 ?
    prev_surface = sitk.ReadImage(prev_surface_path)
    next_surface = sitk.ReadImage(next_surface_path)
    prev_surface = PreProcess(prev_surface)
    next_surface = PreProcess(next_surface)
    prev_size = prev_surface.GetSize()
    next_size = next_surface.GetSize()
    ref_scale = 1
    outside_brightness = 2
    # next_surface = ResizeImg(next_surface, next_size, ref_scale)
    prev_surface = fill_outside(prev_surface, outside_brightness)
    next_surface = fill_outside(next_surface, outside_brightness)
    #   next ?fixed ?prev ?moving??next?
    tp_ = translate_get_align_transform(next_surface, prev_surface,
                                        [os.path.join(PARAMETER_DIR,
                                                      'yq_align_surface_2D.txt')])

    return tp_


def GetOffset(visorPath):
    def Read(file_path):
        # Replace 'path_to_your_file.txt' with the actual path to your text file
        try:
            # Open the text file and read its contents
            with open(file_path, 'r') as file:
                data = file.read()
                # Parse the data as JSON
                json_data = json.loads(data)

                # Extract the specific values
                lefttop_x = json_data['lefttop_x']
                lefttop_y = json_data['lefttop_y']
                lefttop_z = json_data['lefttop_z']

                # Extract the specific values
                rightbottom_x = json_data['rightbottom_x']
                rightbottom_y = json_data['rightbottom_y']
                rightbottom_z = json_data['rightbottom_z']

                # Print the extracted values
                print("lefttop_x:", lefttop_x)
                print("lefttop_y:", lefttop_y)
                print("lefttop_z:", lefttop_z)
                left = [eval(lefttop_x) * 1e3, eval(lefttop_y) * 1e3, float(lefttop_z) * 1e3]
                right = [eval(rightbottom_x) * 1e3, eval(rightbottom_y) * 1e3, float(rightbottom_z) * 1e3]
            return left, right
        except FileNotFoundError:
            print("The file was not found. Please check the path.")
        except json.JSONDecodeError:
            print("Failed to decode JSON. Please check the file content.")
        except KeyError:
            print("One or more keys were not found in the JSON data.")

    # todo get all flsm files
    with open(visorPath) as f:
        info = json.load(f)
    directory_path = os.path.dirname(visorPath)
    acquisition = info['Acquisition Results']
    pathList = []
    leftList = []
    rightList = []
    for flstDict in acquisition:
        temp = flstDict['FlsmList'][0]
        path = os.path.join(directory_path, temp)
        pathList.append(path)
        sliceID = flstDict['SliceID']
        left,right = Read(path)
        leftList.append(left)
        rightList.append(right)
    return leftList , rightList

# todo origin ?bounds ?SliceImage
def SliceResample(imgPath,leftPoint, point, refSize,savePath,checklsPath, checkusPath):

    img = sitk.ReadImage(imgPath)

    imgSize = img.GetSize()
    img.SetSpacing([4,4,4])
    # sliceOrigin = pointsPair[0]
    img.SetOrigin(point)
    newSize = [refSize[0],refSize[1],imgSize[2]]
    refineImg = sitk.Resample(img,newSize,sitk.Transform(),sitk.sitkLinear,leftPoint,[4,4,4])
    # refineImg = sitk.Resample(img,img,sitk.Transform(),sitk.sitkLinear,leftPoint,[4,4,4])
    sitk.WriteImage(refineImg[:,:,175],checklsPath)
    sitk.WriteImage(refineImg[:,:,75],checkusPath)
    # write_ome_tiff(refineImg, savePath)
    pass
# todo ?75 maxprojection
def MaxProjSurface(imgPath, usSavePath, lsSavePath):
    img = sitk.ReadImage(imgPath)
    usIndex = 75
    lsIndex = 175
    maxThickness = 20
    us = sitk.MaximumProjection(img[:,:,usIndex - maxThickness//2 : usIndex + maxThickness//2],projectionDimension=2)[:,:,0]
    ls = sitk.MaximumProjection(img[:,:,lsIndex - maxThickness:lsIndex],projectionDimension=2)[:,:,0]
    sitk.WriteImage(us, usSavePath)
    sitk.WriteImage(ls, lsSavePath)


    return None




import multiprocessing
import time, gc

def step1_1_multiprocess(numsThread, taskParas):
    if numsThread <= 1:
        for task in taskParas:
            taskFun(*task)
        print('All end--')
        return

    pool = multiprocessing.Pool(numsThread)
    result = []
    for i in range(len(taskParas)):
        msg = 'hello %s' % i
        result.append(pool.apply_async(func=taskFun, args=taskParas[i]))

    pool.close()
    pool.join()

    for res in result:
        res.get()

    print('All end--')



def _tiff_size(path):
    with tifffile.TiffFile(path) as tif:
        page_count = len(tif.pages)
        if page_count == 0:
            raise RuntimeError("TIFF has no readable pages: {}".format(path))
        y_size, x_size = tif.pages[0].shape
    return [int(x_size), int(y_size), int(page_count)]


def _read_resampled_tiff_range(path, origin, left_point, ref_size, spacing, z_range):
    z_start, z_end = z_range
    with tifffile.TiffFile(path) as tif:
        page_count = len(tif.pages)
        z_start = max(0, min(int(z_start), page_count))
        z_end = max(z_start + 1, min(int(z_end), page_count))
        data = tif.asarray(key=range(z_start, z_end))
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    image = sitk.GetImageFromArray(data)
    image.SetOrigin([origin[0], origin[1], 0])
    image.SetSpacing(spacing)
    return sitk.Resample(
        image,
        [ref_size[0], ref_size[1], image.GetSize()[2]],
        sitk.Transform(),
        sitk.sitkLinear,
        [left_point[0], left_point[1], 0],
        spacing,
    )


STEP1_1_BLOCK_STATUS_NAME = "step1_1_block_status.txt"
_STEP1_1_BLOCK_STATUS_HEADER = "row\tcol\tstatus\tscore\tup_name\tdown_name\treason\tupdated_at\n"


def _block_save_path(save_root, temp_name, slices_index):
    return os.path.join(save_root, temp_name, "{}_{}".format(slices_index, slices_index + 1))


def _block_file_paths(block_save_path, row_index, col_index):
    prefix = "{}_{}".format(row_index, col_index)
    return (
        os.path.join(block_save_path, prefix + "up_temp_all.tif"),
        os.path.join(block_save_path, prefix + "down_temp_all.tif"),
    )


def _status_file_path(block_save_path):
    return os.path.join(block_save_path, STEP1_1_BLOCK_STATUS_NAME)


def _safe_status_field(value):
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _file_ready(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


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
                "score": parts[3] if len(parts) > 3 else "",
                "up_name": parts[4] if len(parts) > 4 else "",
                "down_name": parts[5] if len(parts) > 5 else "",
                "reason": parts[6] if len(parts) > 6 else "",
                "updated_at": parts[7] if len(parts) > 7 else "",
            }
    return records


def _append_block_status(status_path, row_index, col_index, status, score="", up_name="", down_name="", reason=""):
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    need_header = not os.path.exists(status_path) or os.path.getsize(status_path) == 0
    if score != "":
        score = "{:.6f}".format(float(score))
    updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
        row_index,
        col_index,
        _safe_status_field(status),
        _safe_status_field(score),
        _safe_status_field(up_name),
        _safe_status_field(down_name),
        _safe_status_field(reason),
        updated_at,
    )
    with open(status_path, "a", encoding="utf-8") as file:
        if need_header:
            file.write(_STEP1_1_BLOCK_STATUS_HEADER)
        file.write(line)
        file.flush()


def _block_record_complete(block_save_path, row_index, col_index, record):
    if not record:
        return False
    status = record.get("status")
    if status == "filtered":
        return True
    if status == "saved":
        up_path, down_path = _block_file_paths(block_save_path, row_index, col_index)
        return _file_ready(up_path) and _file_ready(down_path)
    return False


def _step1_1_task_complete(save_root, temp_name, slices_index, ref_size, block_size):
    block_save_path = _block_save_path(save_root, temp_name, slices_index)
    status_path = _status_file_path(block_save_path)
    row = int(np.floor(ref_size[0] / block_size))
    col = int(np.floor(ref_size[1] / block_size))
    if row <= 0 or col <= 0:
        return False
    records = _read_block_statuses(status_path)
    if len(records) < row * col:
        return False
    for row_index in range(row):
        for col_index in range(col):
            if not _block_record_complete(block_save_path, row_index, col_index, records.get((row_index, col_index))):
                return False
    return True


def _bootstrap_step1_1_status_from_existing_outputs(save_root, temp_name, slices_index, ref_size, block_size):
    block_save_path = _block_save_path(save_root, temp_name, slices_index)
    status_path = _status_file_path(block_save_path)
    if os.path.exists(status_path) or not os.path.isdir(block_save_path):
        return False
    row = int(np.floor(ref_size[0] / block_size))
    col = int(np.floor(ref_size[1] / block_size))
    if row <= 0 or col <= 0:
        return False

    existing_count = 0
    for row_index in range(row):
        for col_index in range(col):
            save_name1, save_name2 = _block_file_paths(block_save_path, row_index, col_index)
            if _file_ready(save_name1) and _file_ready(save_name2):
                existing_count += 1
    if existing_count == 0:
        return False

    for row_index in range(row):
        for col_index in range(col):
            save_name1, save_name2 = _block_file_paths(block_save_path, row_index, col_index)
            if _file_ready(save_name1) and _file_ready(save_name2):
                _append_block_status(
                    status_path,
                    row_index,
                    col_index,
                    "saved",
                    "",
                    os.path.basename(save_name1),
                    os.path.basename(save_name2),
                    "legacy_existing_output",
                )
            else:
                _append_block_status(
                    status_path,
                    row_index,
                    col_index,
                    "filtered",
                    "",
                    "",
                    "",
                    "legacy_assumed_filtered_no_tiff",
                )
    print(
        "Step1_1 bootstrapped status from existing outputs: {} saved={} filtered={}".format(
            status_path,
            existing_count,
            row * col - existing_count,
        )
    )
    return True


def step1_1_task_complete(save_root, temp_name, slices_index, ref_size, block_size):
    return _step1_1_task_complete(save_root, temp_name, slices_index, ref_size, block_size)


def taskFun(up_path, down_path, upOrigin, downOrigin, left_point, refSize, spacing, i,
            save_root, temp_name='temp_block', block_size=250):
    # 

    # ?
    print(f"Reconstruction started for data chunk ")
    print(f"String input: {up_path, down_path}")


    # print(f"Reconstruction completed for data chunk {data_id}")
    print("left_point is : ", left_point)

    block_save_path = _block_save_path(save_root, temp_name, i)
    _bootstrap_step1_1_status_from_existing_outputs(save_root, temp_name, i, refSize, block_size)
    if _step1_1_task_complete(save_root, temp_name, i, refSize, block_size):
        print("Step1_1 slice {} already complete from status file: {}".format(i, block_save_path))
        return

    up_size = _tiff_size(up_path)
    down_size = _tiff_size(down_path)
    interval = 60
    gap = 10
    first = up_size[2] - gap
    second = down_size[2] - gap - 100 + 10
    up_roi = [first - interval, first]
    down_roi = [second - interval, second]
    up_img = _read_resampled_tiff_range(up_path, upOrigin, left_point, refSize, spacing, up_roi)
    down_img = _read_resampled_tiff_range(down_path, downOrigin, left_point, refSize, spacing, down_roi)

    print("down_img.GetSpacing() : {}\n Origin: {} \n Size: {}".format(down_img.GetSpacing(), down_img.GetOrigin(),
                                                                       down_img.GetSize()))
    print("up_img.GetSpacing() : {}\n Origin: {}\n Size: {}".format(up_img.GetSpacing(), up_img.GetOrigin(),
                                                                    up_img.GetSize()))
    temp_img_path = os.path.join(save_root,"temp_img")
    os.makedirs(temp_img_path, exist_ok=True)
    # sitk.WriteImage(up_img[:,:,75],os.path.join(temp_img_path,"{}_75.tif".format(i)))
    # sitk.WriteImage(up_img[:,:,175],os.path.join(temp_img_path,"{}_175.tif".format(i)))
    # todo  xy 
    start = time.time()

    overlap_depth = min(up_img.GetSize()[2], down_img.GetSize()[2])
    roi = [[0, overlap_depth], [0, overlap_depth]]

    next_result = None
    print("Coarse alignment elapsed time: {}".format(time.time() - start))

    split_block(next_result, up_img, down_img, spacing,  roi=roi, slices_index=i
                  ,save_root = save_root
                  ,tempName = temp_name,
                  block_size=block_size,
                  sub_block=block_size)

    print("the space of {} cost : {} ".format(i, time.time() - start))
    gc.collect()

def split_block(img, up_img, down_img, spacing, roi, slices_index,
                  block_size=250, sub_block=250, save_root=r"D:\USERS\yq\code\cal_overlap\Refine",
                  tempName='th2_111_112'):
    if img == None:
        img = sitk.MaximumProjection(down_img, projectionDimension=2)[:, :, 0]
    up_img.SetSpacing(spacing)
    up_img.SetOrigin([0, 0, 0])
    down_img.SetSpacing(spacing)
    down_img.SetOrigin([0, 0, 0])
    size = img.GetSize()
    row = int(np.floor(size[0] / block_size))
    col = int(np.floor(size[1] / block_size))
    vector_points = np.zeros((row, col, 3))
    forbid_points = np.zeros((row, col))
    # todo ??ban?
    back_brightness = 120
    for i in range(row):
        for j in range(col):
            temp = sitk.GetArrayFromImage(img[i * block_size: (i + 1) * block_size,
                                          j * block_size:(j + 1) * block_size])
            sub_temp = temp[block_size - sub_block:, block_size - sub_block:]
            bool_temp = sub_temp > back_brightness
            int_temp = np.array(bool_temp, dtype=np.int32)

            holow_scale = np.mean(np.mean(int_temp))
            forbid_points[i, j] = holow_scale

            # print("")
    # print("")

    # todo ?
    # up_img = Preprocess(up_img, 120)
    # down_img = Preprocess(down_img, 120)

    tf_pars = []
    pos = []
    # i,j = 5,2
    all_blocks = [
        (i, j)
        for i in range(row)
        for j in range(col)
    ]
    selected_blocks = [
        (i, j)
        for i, j in all_blocks
        if forbid_points[i, j] > 0.4
    ]
    used_fallback = False
    if not selected_blocks:
        flat_scores = [
            (forbid_points[i, j], i, j)
            for i, j in all_blocks
        ]
        flat_scores.sort(reverse=True)
        fallback_count = min(64, len(flat_scores))
        selected_blocks = [(i, j) for score, i, j in flat_scores[:fallback_count]]
        used_fallback = True
        print("No blocks passed foreground threshold for slice {}. Using {} strongest blocks.".format(
            slices_index,
            len(selected_blocks),
        ))

    os.makedirs(os.path.join(save_root, tempName), exist_ok=True)
    block_save_path = _block_save_path(save_root, tempName, slices_index)
    if not os.path.exists(block_save_path):
        os.mkdir(block_save_path)
    print("Step1_1 block output folder: {}".format(block_save_path))

    status_path = _status_file_path(block_save_path)
    records = _read_block_statuses(status_path)
    selected_set = set(selected_blocks)
    filtered_reason = "fallback_not_selected" if used_fallback else "foreground_below_threshold"

    for row_index, col_index in all_blocks:
        if (row_index, col_index) in selected_set:
            continue
        save_name1, save_name2 = _block_file_paths(block_save_path, row_index, col_index)
        if _file_ready(save_name1) and _file_ready(save_name2):
            if records.get((row_index, col_index), {}).get("status") != "saved":
                _append_block_status(
                    status_path,
                    row_index,
                    col_index,
                    "saved",
                    forbid_points[row_index, col_index],
                    os.path.basename(save_name1),
                    os.path.basename(save_name2),
                    "existing_output",
                )
            continue
        if records.get((row_index, col_index), {}).get("status") != "filtered":
            _append_block_status(
                status_path,
                row_index,
                col_index,
                "filtered",
                forbid_points[row_index, col_index],
                "",
                "",
                filtered_reason,
            )

    for i, j in selected_blocks:
        start = time.time()
        save_name1, save_name2 = _block_file_paths(block_save_path, i, j)
        if _file_ready(save_name1) and _file_ready(save_name2):
            if records.get((i, j), {}).get("status") != "saved":
                _append_block_status(
                    status_path,
                    i,
                    j,
                    "saved",
                    forbid_points[i, j],
                    os.path.basename(save_name1),
                    os.path.basename(save_name2),
                    "existing_output",
                )
            continue
        up_temp = up_img[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size,
                  roi[0][0]:roi[0][1]]
        down_temp = down_img[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size,
                    roi[1][0]:roi[1][1]]
        sub_up = up_temp[block_size - sub_block:, block_size - sub_block:, :]
        sub_down = down_temp[block_size - sub_block:, block_size - sub_block:, :]

        origin = [0, 0, 0]
        sub_up.SetOrigin(origin)
        sub_up.SetSpacing(spacing)
        sub_down.SetOrigin(origin)
        sub_down.SetSpacing(spacing)

        write_ome_tiff(sub_up, save_name1)
        write_ome_tiff(sub_down, save_name2)
        _append_block_status(
            status_path,
            i,
            j,
            "saved",
            forbid_points[i, j],
            os.path.basename(save_name1),
            os.path.basename(save_name2),
            "written",
        )

    final_records = _read_block_statuses(status_path)
    saved_count = sum(1 for record in final_records.values() if record.get("status") == "saved")
    filtered_count = sum(1 for record in final_records.values() if record.get("status") == "filtered")
    print(
        "Step1_1 block status file: {} saved={} filtered={} total={}".format(
            status_path,
            saved_count,
            filtered_count,
            row * col,
        )
    )

def ReadNPY():
    a = np.load("Refine/tf_155_pars.npy")
    a_0 = a[:,:,0]
    a_1 = a[:,:,1]
    a_2 = a[:,:,2]
    print()
def main():
    # todo read reconstruction info and use the point bounds to resample the size of the image
    imgFormat = r"E:\20250426_SMY_TAC1_AI14_1_1\Reconstruction\SliceImage\4.0\TAC1_AI14_1_{:03d}_561nm_10X.tif"

    # TODO  ?
    taskChunk = []
    # visorPath = r"E:\CRH_all.visor"
    visorPath = r"E:\tac1_ai14.visor"
    leftList, rightList = GetOffset(visorPath)
    leftList = np.array(leftList)
    rightList = np.array(rightList)
    num = len(leftList)

    spacing = [4,4,4]
    lefttop = leftList.min(axis=0)
    rightbottom = rightList.max(axis=0)
    lefttop = [lefttop[0], lefttop[1], 0]
    refSize = [(rightbottom[0] - lefttop[0]) // spacing[0], (rightbottom[1] - lefttop[1]) // spacing[1]]
    refSize = [int(i) for i in refSize]
    refSize = [10500,6750]
    npy_format = "tf_{}_pars.npy"
    temp_root = r"E:\20250426_SMY_TAC1_AI14_1_1\Reconstruction\tac"
    os.makedirs(temp_root,exist_ok=True)
    for i in range(36, 42):
        # if os.path.exists(os.path.join(temp_root, npy_format.format(i))) :
        #     print(f"exist {npy_format.format(i)}")
        #     continue
        prevIndex = i
        nextIndex = i + 1
        upOrigin = leftList[prevIndex - 1]
        upOrigin[2] = 0
        downOrigin = leftList[nextIndex - 1]
        downOrigin[2] = 0
        up_path = imgFormat.format(prevIndex)
        down_path = imgFormat.format(nextIndex)
        # bottom1 = heightPairs[i][1]
        temp = (up_path, down_path, upOrigin, downOrigin, lefttop, refSize, spacing, i, temp_root)
        taskChunk.append(temp)
        # taskFun(up_path, down_path, upOrigin, downOrigin, lefttop, refSize, spacing, i,temp_root)

    num_threads = 8  # 
    run_multiprocess(num_threads, taskChunk)

if __name__ == '__main__':
    main()
