"""B85 refinement step 1.1 helpers.

This module creates adjacent-slice image blocks used for coarse alignment.
It is a local copy of the B85 utility code so the runner does not depend on
YQReconstructionScripts at runtime.
"""
from .yq_elastix_files import *
import numpy as np
import os

import unittest
from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
from VISoR_Brain.utils.elastix_files import *
from VISoR_Reconstruction.reconstruction.brain_reconstruct_methods.common import fill_outside
from .common0313 import *

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
    # todo use multiprocess
    pool = multiprocessing.Pool(numsThread)
    result = []
    for i in range(len(taskParas)):
        msg = 'hello %s' % i
        result.append(pool.apply_async(func=taskFun, args=taskParas[i]))

    pool.close()
    pool.join()

    # for res in result:
    #     print('***:', res.get())  # get()?

    print('All end--')



def taskFun(up_path, down_path, upOrigin, downOrigin, left_point, refSize, spacing, i,
            save_root):
    # 

    # ?
    print(f"Reconstruction started for data chunk ")
    print(f"String input: {up_path, down_path}")


    # print(f"Reconstruction completed for data chunk {data_id}")
    up_img = sitk.ReadImage(up_path)
    down_img = sitk.ReadImage(down_path)

    #  ?
    # todo  ?
    # left_point = [0,0,0]
    print("left_point is : ", left_point)
    # todo

    # todo ??
    up_img.SetOrigin(upOrigin)
    up_img.SetSpacing(spacing)
    down_img.SetOrigin(downOrigin)
    down_img.SetSpacing(spacing)
    # init transform
    dimension = 3
    up_size = up_img.GetSize()
    up_img = sitk.Resample(up_img, [refSize[0], refSize[1], up_size[2]],
                             sitk.Transform(), sitk.sitkLinear, left_point, spacing)

    down_size = down_img.GetSize()
    down_img = sitk.Resample(down_img, [refSize[0], refSize[1], down_size[2]],
                             sitk.Transform(), sitk.sitkLinear, left_point, spacing)
    # sitk.WriteImage()

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

    #   2D data ?
    size1 = up_img.GetSize()
    size2 = down_img.GetSize()
    #  ?
    # bottom1 = GetBottom_4um(size1)
    # bottom2 = GetBottom_4um(size2)
    # end2 = int(bottom2 - 40 * 2.5)
    interval = 60
    gap = 10
    first = up_size[2] - gap
    second = down_size[2] - gap - 100 + 10
    roi = [[first - interval,first], [second - interval,second]]
    # todo  4 ?
    # spacing = [4,4,4]

    next_result = None
    print("Coarse alignment elapsed time: {}".format(time.time() - start))

    split_block(next_result, up_img, down_img, spacing,  roi=roi, slices_index=i
                  ,save_root = save_root
                  ,tempName = 'temp_block')

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
    for i in range(row):
        for j in range(col):

            if forbid_points[i, j] > 0.4:
                start = time.time()
                up_temp = up_img[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size,
                          roi[0][0]:roi[0][1]]
                down_temp = down_img[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size,
                            roi[1][0]:roi[1][1]]
                sub_up = up_temp[block_size - sub_block:, block_size - sub_block:, :]
                sub_down = down_temp[block_size - sub_block:, block_size - sub_block:, :]

                # todo 
                max_sub_down = sitk.MaximumProjection(sub_down[:, :, :(roi[0][1] - roi[0][0]) // 2],
                                                      projectionDimension=2)[:, :, 0]
                hollow_scale = np.mean(np.mean(max_sub_down))
                if hollow_scale < 0.4:
                    continue

                # todo  start  

                # todo  end

                origin = [0, 0, 0]
                # todo  200 * 200 ??
                sub_up.SetOrigin(origin)
                sub_up.SetSpacing(spacing)
                sub_down.SetOrigin(origin)
                sub_down.SetSpacing(spacing)
                # create file folder
                os.makedirs(os.path.join(save_root, tempName), exist_ok=True)

                block_save_path = os.path.join(save_root, tempName, str(slices_index) + '_' + str(slices_index + 1))
                if not os.path.exists(block_save_path):
                    os.mkdir(block_save_path)
                save_name1 = os.path.join(block_save_path, str(i) + "_" + str(j) + "up_temp_all.tif")
                save_name2 = os.path.join(block_save_path, str(i) + "_" + str(j) + "down_temp_all.tif")
                # sitk.WriteImage(sub_up, save_name1)
                # sitk.WriteImage(sub_down, save_name2)
                if os.path.exists(save_name1) and os.path.exists(save_name2):
                    continue
                write_ome_tiff(sub_up, save_name1)
                write_ome_tiff(sub_down, save_name2)

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
