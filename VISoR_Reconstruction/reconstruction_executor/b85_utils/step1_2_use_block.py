"""B85 refinement step 1.2 helpers.

This module aligns adjacent image blocks and writes coarse position results.
It is stored inside VISoR_Reconstruction to avoid runtime imports from
YQReconstructionScripts.
"""
from .yq_elastix_files import *
import ast
import numpy as np
import os

import unittest
from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
from VISoR_Brain.utils.elastix_files import *
from VISoR_Reconstruction.reconstruction.brain_reconstruct_methods.common import fill_outside
from .common0313 import Preprocess, fill_outside_yq


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

import multiprocessing
import time, gc

STEP1_2_BLOCK_STATUS_NAME = "step1_2_block_status.txt"
_STEP1_2_BLOCK_STATUS_HEADER = "row\tcol\tstatus\tpos_name\treason\tmessage\tupdated_at\n"


def _safe_status_field(value):
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _file_ready(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _pos_file_path(save_res_folder, row_index, col_index):
    return os.path.join(save_res_folder, f"pos_{row_index}_{col_index}.txt")


def _status_file_path(save_res_folder):
    return os.path.join(save_res_folder, STEP1_2_BLOCK_STATUS_NAME)


def _pos_file_ready(path):
    if not _file_ready(path):
        return False
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as file:
            value = ast.literal_eval(file.read().strip())
        return isinstance(value, (list, tuple)) and len(value) == 3
    except (OSError, SyntaxError, ValueError):
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
                "pos_name": parts[3] if len(parts) > 3 else "",
                "reason": parts[4] if len(parts) > 4 else "",
                "message": parts[5] if len(parts) > 5 else "",
                "updated_at": parts[6] if len(parts) > 6 else "",
            }
    return records


def _append_block_status(status_path, row_index, col_index, status, pos_name="", reason="", message=""):
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    need_header = not os.path.exists(status_path) or os.path.getsize(status_path) == 0
    updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
        row_index,
        col_index,
        _safe_status_field(status),
        _safe_status_field(pos_name),
        _safe_status_field(reason),
        _safe_status_field(message),
        updated_at,
    )
    with open(status_path, "a", encoding="utf-8") as file:
        if need_header:
            file.write(_STEP1_2_BLOCK_STATUS_HEADER)
        file.write(line)
        file.flush()


def _append_task_result(result):
    if result is None:
        return
    save_res_folder, row_index, col_index, status, pos_name, reason, message = result
    _append_block_status(
        _status_file_path(save_res_folder),
        row_index,
        col_index,
        status,
        pos_name,
        reason,
        message,
    )


def _block_record_complete(save_res_folder, row_index, col_index, record):
    if not record:
        return False
    status = record.get("status")
    if status == "aligned":
        return _pos_file_ready(_pos_file_path(save_res_folder, row_index, col_index))
    if status in ("failed", "skipped_no_step1_1_block"):
        return True
    return False


def step1_2_block_complete(save_res_folder, row_index, col_index):
    records = _read_block_statuses(_status_file_path(save_res_folder))
    return _block_record_complete(save_res_folder, row_index, col_index, records.get((row_index, col_index)))


def bootstrap_step1_2_status(save_res_folder, expected_blocks=None, all_blocks=None):
    os.makedirs(save_res_folder, exist_ok=True)
    expected_blocks = set(expected_blocks or [])
    all_blocks = set(all_blocks or expected_blocks)
    status_path = _status_file_path(save_res_folder)
    records = _read_block_statuses(status_path)
    changed = False

    for row_index, col_index in sorted(all_blocks):
        pos_path = _pos_file_path(save_res_folder, row_index, col_index)
        record = records.get((row_index, col_index))
        if (row_index, col_index) in expected_blocks:
            if _pos_file_ready(pos_path):
                if not record or record.get("status") != "aligned":
                    _append_block_status(
                        status_path,
                        row_index,
                        col_index,
                        "aligned",
                        os.path.basename(pos_path),
                        "legacy_existing_pos",
                    )
                    changed = True
            continue

        if not record or record.get("status") != "skipped_no_step1_1_block":
            _append_block_status(
                status_path,
                row_index,
                col_index,
                "skipped_no_step1_1_block",
                "",
                "step1_1_filtered_or_missing",
            )
            changed = True
    return changed


def step1_2_pair_complete(save_res_folder, expected_blocks):
    records = _read_block_statuses(_status_file_path(save_res_folder))
    for row_index, col_index in expected_blocks:
        if not _block_record_complete(save_res_folder, row_index, col_index, records.get((row_index, col_index))):
            return False
    return True


def step1_2_incomplete_blocks(save_res_folder, expected_blocks):
    records = _read_block_statuses(_status_file_path(save_res_folder))
    incomplete = []
    for row_index, col_index in sorted(expected_blocks):
        record = records.get((row_index, col_index))
        if _block_record_complete(save_res_folder, row_index, col_index, record):
            continue
        pos_path = _pos_file_path(save_res_folder, row_index, col_index)
        incomplete.append(
            {
                "row": row_index,
                "col": col_index,
                "status": record.get("status", "missing") if record else "missing",
                "pos_ready": _pos_file_ready(pos_path),
            }
        )
    return incomplete


def finalize_step1_2_status(save_res_folder, expected_blocks, all_blocks=None, mark_missing_failed=False):
    bootstrap_step1_2_status(save_res_folder, expected_blocks, all_blocks)
    incomplete = step1_2_incomplete_blocks(save_res_folder, expected_blocks)
    if mark_missing_failed:
        status_path = _status_file_path(save_res_folder)
        for block in incomplete:
            if block["pos_ready"]:
                _append_block_status(
                    status_path,
                    block["row"],
                    block["col"],
                    "aligned",
                    os.path.basename(_pos_file_path(save_res_folder, block["row"], block["col"])),
                    "recovered_existing_pos",
                    "status file was repaired after multiprocessing",
                )
            else:
                _append_block_status(
                    status_path,
                    block["row"],
                    block["col"],
                    "failed",
                    "",
                    "missing_pos_after_step1_2",
                    "pos file was not generated by the worker",
                )
        bootstrap_step1_2_status(save_res_folder, expected_blocks, all_blocks)
        incomplete = step1_2_incomplete_blocks(save_res_folder, expected_blocks)
    return incomplete


def step1_2_multiprocess(numsThread, taskParas):
    # todo use multiprocess
    if not taskParas:
        print('No step1_2 block tasks to run--')
        return
    if numsThread <= 1:
        for task in taskParas:
            _append_task_result(deal_block(*task))
        print('All end--')
        return

    pool = multiprocessing.Pool(numsThread)
    result = []
    for i in range(len(taskParas)):
        msg = 'hello %s' % i
        result.append(pool.apply_async(func=deal_block, args=taskParas[i]))
    pool.close()
    pool.join()

    for res in result:
        _append_task_result(res.get())

    print('All end--')




def get_block_name(refSize,slices_index,
                  block_size, save_root,
                  tempName):
    name_list = []
    size = refSize
    row = int(np.floor(size[0] / block_size))
    col = int(np.floor(size[1] / block_size))
    for i in range(row):
        for j in range(col):
            block_save_path = os.path.join(save_root, tempName, str(slices_index) + '_' + str(slices_index + 1))

            save_name1 = os.path.join(block_save_path, str(i) + "_" + str(j) + "up_temp_all.tif")
            save_name2 = os.path.join(block_save_path, str(i) + "_" + str(j) + "down_temp_all.tif")
            if os.path.exists(save_name1) and os.path.exists(save_name2):
                name_list.append([save_name1, save_name2])
    return name_list
            # sitk.WriteImage(sub_up, save_name1)
            # sitk.WriteImage(sub_down, save_name2)

'''
save_name1 = os.path.join(block_save_path, str(i)+"_"+str(j)+"up_temp_all.tif")
save_name2 = os.path.join(block_save_path, str(i)+"_"+str(j)+"down_temp_all.tif")
sitk.WriteImage(sub_up, save_name1)
sitk.WriteImage(sub_down, save_name2)
# write_ome_tiff(sub_up, save_name1)
# write_ome_tiff(sub_down, save_name2)

# prev_surface, next_result, transform2 = CalBlock(sub_up, sub_down, spacing)
try:
    prev_surface, next_result, transform2 = CalBlock(sub_up,sub_down,spacing)
    print(transform2)
    param = transform2.GetParameters()
    # param[2] = param[2] + drop
    vector_points[i, j, :] = np.array(param)
    tf_pars.append(param)
    pos.append([i,j])
    # todo tf pars   next img ?
    with open(os.path.join(save_root, tempName,'tf_'+str(slices_index)+'_pars.txt'), 'w') as file:
        # ?
        for k in range(len(tf_pars)):
            file.write(str(pos[k])+": " + str(tf_pars[k]) + '\n')
    # if tf_pars is not None:
    #     print("tf_pars is : ",tf_pars)
    #     break
    print("{} {} costs time :{}".format(i,j,time.time()-start))
    print()
except:
    print("row: {}; col: {} gets wrong!!!".format(i,j))
    continue

'''
def ReadNPY():
    a = np.load("Refine/tf_155_pars.npy")
    a_0 = a[:,:,0]
    a_1 = a[:,:,1]
    a_2 = a[:,:,2]
    print()
def deal_block(up_path, donw_path, i,j,save_res_folder):
    save_path = _pos_file_path(save_res_folder, i, j)
    status_path = _status_file_path(save_res_folder)
    print(save_path)
    if step1_2_block_complete(save_res_folder, i, j):
        print(f"Step1_2 block {i}_{j} already complete from status file: {status_path}")
        return None

    start = time.time()
    try:
        sub_up = sitk.ReadImage(up_path)[:,:,10:]
        sub_down = sitk.ReadImage(donw_path)[:,:,:-10]
        threshold = 120
        # ?
        pixel_type = sub_up.GetPixelID()

        #
        if pixel_type == sitk.sitkUInt8:
            print("Image pixel type is sitkUInt8; preprocessing is skipped.")
        elif pixel_type == sitk.sitkUInt16:
            print("Image pixel type is sitkUInt16; preprocessing is applied.")
            sub_up = Preprocess(sub_up, threshold)

        pixel_type = sub_down.GetPixelID()
        #
        if pixel_type == sitk.sitkUInt8:
            print("Image pixel type is sitkUInt8; preprocessing is skipped.")
        elif pixel_type == sitk.sitkUInt16:
            print("Image pixel type is sitkUInt16; preprocessing is applied.")
            sub_down = Preprocess(sub_down, threshold)
        spacing = [4,4,4]
        origin = [0,0,0]
        sub_up.SetOrigin(origin)
        sub_up.SetSpacing(spacing)
        sub_down.SetOrigin(origin)
        sub_down.SetSpacing(spacing)


        # write_ome_tiff(sub_up, save_name1)
        # write_ome_tiff(sub_down, save_name2)

        # prev_surface, next_result, transform2 = CalBlock(sub_up, sub_down, spacing)
        prev_surface, next_result, transform2 = CalBlock(sub_up, sub_down, spacing)
        print(transform2)
        param = transform2.GetParameters()
        print(f"param is {param}")
        # txt
        with open(save_path, "w") as f:
            f.write(str(param))
        result = (
            save_res_folder,
            i,
            j,
            "aligned",
            os.path.basename(save_path),
            "written",
            "",
        )
        print("{} {} costs time :{}".format(i, j, time.time() - start))
        # sitk.WriteImage(prev_surface, r"D:\USERS\yq\CRH\save_temp_0413\temp_block\prev_surface.tif")
        # sitk.WriteImage(next_result, r"D:\USERS\yq\CRH\save_temp_0413\temp_block\next_result.tif")
        print()
        return result
    except Exception as exc:
        print("row: {}; col: {} gets wrong!!! {}".format(i, j, exc))
        return (
            save_res_folder,
            i,
            j,
            "failed",
            "",
            exc.__class__.__name__,
            str(exc),
        )
def CalBlock(prev_surface, next_surface,spacing, ref_img: sitk.Image = None, prev_points=None, next_points=None,
                   outside_brightness=2, nonrigid=True, ref_size=None, ref_scale=1, use_rigidity_mask=False, **kwargs):
    size = prev_surface.GetSize()
    temp = sitk.GetArrayFromImage(prev_surface)
    if len(size) == 2:
        temp[:, :] = fill_outside_yq(temp[:, :], outside_brightness)
    else:
        for i in range(size[2]):
            temp[i, :, :] = fill_outside_yq(temp[i, :, :], outside_brightness)
    prev_surface = sitk.GetImageFromArray(temp)

    size = next_surface.GetSize()
    temp = sitk.GetArrayFromImage(next_surface)
    if len(size) == 2:
        temp[:, :] = fill_outside_yq(temp[:, :], outside_brightness)
    else:
        for i in range(size[2]):
            temp[i, :, :] = fill_outside_yq(temp[i, :, :], outside_brightness)
    next_surface = sitk.GetImageFromArray(temp)

    '''
    ?affine 
    '''
    # justify the spcaing of imgs
    prev_surface.SetSpacing(spacing)
    next_surface.SetSpacing(spacing)

    next_result, transform2 = get_align_transform(prev_surface, next_surface,
                                                  [os.path.join(PARAMETER_DIR, 'p_3D_250320.txt')])
    return prev_surface, next_result, transform2

def copy_extract_surface(img: sitk.Image, umap: sitk.Image, lmap: sitk.Image):
    img.SetSpacing([1, 1, 1])
    img.SetOrigin([0, 0, 0])
    umap.SetSpacing([1, 1, 1])
    umap.SetOrigin([0, 0, 0])
    lmap.SetSpacing([1, 1, 1])
    lmap.SetOrigin([0, 0, 0])

    # umap_s = umap + 1
    # lmap_s = lmap - 1
    # zeros = sitk.Image(umap.GetSize(), umap.GetPixelIDValue())
    # df = sitk.JoinSeries(sitk.Compose(zeros, zeros, umap_s), sitk.Compose(zeros, zeros, lmap_s))

    df = sitk.JoinSeries(umap, lmap)

    df = sitk.Cast(df, sitk.sitkVectorFloat64)
    ref = sitk.Image(df)
    tr = sitk.DisplacementFieldTransform(3)
    tr.SetDisplacementField(df)
    surfaces = sitk.Resample(img, ref, tr)
    # surfaces = sitk.Cast(surfaces, sitk.sitkFloat32)
    # surfaces = sitk.Clamp((sitk.Log(sitk.Cast(surfaces, sitk.sitkFloat32)) - 4.6) * 39.4, sitk.sitkUInt8, 0, 255)
    # surfaces = sitk.Cast(surfaces, sitk.sitkFloat32)
    return surfaces
def extract_surface_failed(name_format):
    # img_size = [6500,5500]
    # img_size = [5609, 5517]
    # img_size = refSize
    # temp_root = save_root
    print(f"refSize is {refSize}")


    uz_name_format = os.path.join(temp_root, name_format + "_uz.mha")
    lz_name_format = os.path.join(temp_root, name_format + "_lz.mha")

    us_name_format = os.path.join(temp_root, name_format + "_us.mha")
    ls_name_format = os.path.join(temp_root, name_format + "_ls.mha")
    n_start = 0
    spacing = [4,4,4]
    range_ = [i for i in range(1,197)]
    for i in range_:
        uz_path = uz_name_format.format(i)
        lz_path = lz_name_format.format(i)
        if os.path.exists(uz_path) and os.path.exists(lz_path):
            print(f"{uz_path} exists")
            continue
        # todo ??
        imgOrigin = leftList[i - 1]
        imgOrigin = [imgOrigin[0], imgOrigin[1], 0]
        print(f"imgOrigin is {imgOrigin} lefttop is {lefttop}")

        img = sitk.ReadImage(imgFormat.format(i))
        img.SetOrigin(imgOrigin)
        img.SetSpacing(spacing)
        img_size = img.GetSize()
        print(f" img size is {img_size}")

        # sitk.WriteImage(img[:,:,302],os.path.join(temp_root, "312.tif"))
        # sitk.WriteImage(img[:,:,202],os.path.join(temp_root, "212.tif"))
        # init transform
        dimension = 3
        img.SetSpacing(spacing)
        img = sitk.Resample(img, [refSize[0], refSize[1], img_size[2]],
                               sitk.Transform(), sitk.sitkLinear, lefttop, spacing)
        gap = 10
        height_range = [img_size[2] - 100 - gap, img_size[2] - gap]

        # sitk.WriteImage(img[:,:,-100 - gap],os.path.join(temp_root, "312.tif"))
        # sitk.WriteImage(img[:,:,0 - gap],os.path.join(temp_root, "212.tif"))
        img.SetOrigin([0,0,0])
        img.SetSpacing([1,1,1])

        print(uz_path)
        print(lz_path)
        ### todo surface Displace
        umap_x = sitk.Image(refSize, sitk.sitkFloat32)
        umap_y = sitk.Image(refSize, sitk.sitkFloat32)
        # umap_z = sitk.ReadImage(uz_path)
        umap_z = sitk.Image(refSize, sitk.sitkFloat32) + height_range[0]
        uz = sitk.Compose(umap_x, umap_y, umap_z)

        lmap_x = sitk.Image(refSize, sitk.sitkFloat32)
        lmap_y = sitk.Image(refSize, sitk.sitkFloat32)
        lmap_z = sitk.Image(refSize, sitk.sitkFloat32) + height_range[1]
        lz = sitk.Compose(lmap_x, lmap_y, lmap_z)

        surfaces = copy_extract_surface(img, uz, lz)

        us = surfaces[:, :, 0]
        ls = surfaces[:, :, 1]

        sitk.WriteImage(uz, uz_name_format.format(i))
        sitk.WriteImage(lz, lz_name_format.format(i))

        sitk.WriteImage(us, us_name_format.format(i))
        sitk.WriteImage(ls, ls_name_format.format(i))


if __name__ == '__main__':
    visorPath = r"E:\tac1_ai14.visor"
    leftList, rightList = GetOffset(visorPath)
    leftList = np.array(leftList)
    rightList = np.array(rightList)

    spacing = [4, 4, 4]
    lefttop = leftList.min(axis=0)
    rightbottom = rightList.max(axis=0)
    lefttop = [lefttop[0], lefttop[1], 0]
    refSize = [(rightbottom[0] - lefttop[0]) // spacing[0], (rightbottom[1] - lefttop[1]) // spacing[1]]
    refSize = [int(i) for i in refSize]
    # refSize = [10750, 6000]
    refSize = [10500,6750]

    npy_format = "tf_{}_pars.npy"
    re_npy_format = "refine_{}_pars.npy"
    save_root = r"E:\20250426_SMY_TAC1_AI14_1_1\Reconstruction\tac"
    temp_root = r"E:\20250426_SMY_TAC1_AI14_1_1\Reconstruction\tac\temp_block"
    save_res_root = r"E:\20250426_SMY_TAC1_AI14_1_1\Reconstruction\tac\minus10"
    os.makedirs(save_res_root,exist_ok=True)

    main()

    # uzlzRoot = temp_root
    # # uzlzRoot = r"K:\STZ1_914#\save_temp_0313\Temp"
    # name_format = "TAC1_AI14_1_{:03d}_561nm_10X"
    # imgFormat = r"E:\20250426_SMY_TAC1_AI14_1_1\Reconstruction\SliceImage\4.0\TAC1_AI14_1_{:03d}_561nm_10X.tif"
    #
    # extract_surface()
