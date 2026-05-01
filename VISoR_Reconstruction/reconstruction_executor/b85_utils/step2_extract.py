"""B85 refinement step 2 helpers.

This module extracts upper/lower z and surface images from refine parameters.
"""

import json
import os.path
import unittest
from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
from VISoR_Brain.utils.elastix_files import *
from VISoR_Reconstruction.reconstruction.brain_reconstruct_methods.common import fill_outside
from .common0424 import *

def ReadOffsetTxt(txtPath = r"D:\USERS\yq\code\cal_overlap\Refine\th2_33\tf_33_pars.txt"):
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


def CreateTransform(x, y, z, img_size):
    from scipy.interpolate import RectBivariateSpline

    func = RectBivariateSpline(x, y, z, s=0)

    # xnew = np.arange(0, shape[0], 1e-1)
    # ynew = np.arange(0, shape[1], 1e-1)
    # xnew = np.arange(0, img_size[0], 1e-3)
    # ynew = np.arange(0, img_size[1], 1e-3)
    xnew = np.arange(0, img_size[0], 1)
    ynew = np.arange(0, img_size[1], 1)
    znew = func(xnew, ynew)
    return xnew, ynew, znew

def CreateTransform_Test(x, y, z, img_size):
    original = z
    block_size = 250
    # block_size = 3  #  block_size ?

    #  np.kron ?original ?block_size x block_size 
    deformation_field = np.kron(original, np.ones((block_size, block_size)))

    print(deformation_field.shape)  #  (43*block_size, 24*block_size)

    return None, None, deformation_field

def Flip(img,affine_t):


    af = sitk.AffineTransform(2)
    af.SetMatrix(affine_t)
    size = img.GetSize()
    size = [size[1], size[0]]
    sitk_image = sitk.Resample(img, size, af)
    # write_ome_tiff(sitk_image,'temp.tif')
    return sitk_image

# ?cal overlap
def ReadNpy(txtPath,tempName):
    import re

    # 
    offsets = {}

    # 
    # txtPath = r"D:\USERS\yq\code\cal_overlap\Refine\th2_0511\tf_33_pars.txt"
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
    # tempName = 'th2_0511/33_34'
    spacing = [4.0, 4.0, 4.0]
    # spacing = [1.0, 1.0, 1.0]
    for key, value in offsets.items():
        print(f"key is {key}; value is {value}")
        i = key[0];
        j = key[1];
        fixed = sitk.ReadImage(os.path.join(r"D:\USERS\yq\code\cal_overlap\Refine", tempName,
                                            str(i) + "_" + str(j) + "up_temp_all.tif"))
        moving = sitk.ReadImage(os.path.join(r"D:\USERS\yq\code\cal_overlap\Refine", tempName,
                                             str(i) + "_" + str(j) + "down_temp_all.tif"))
        movedPath = (os.path.join(r"D:\USERS\yq\code\cal_overlap\Refine", tempName,
                                  str(i) + "_" + str(j) + "moved.tif"))

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
        sitk.WriteImage(resampled_image, movedPath)
import multiprocessing
import time, gc

def step2_multiprocess(numsThread, taskParas):
    # todo use multiprocess
    pool = multiprocessing.Pool(numsThread)
    result = []
    for i in range(len(taskParas)):
        msg = 'hello %s' % i
        result.append(pool.apply_async(func=MainTask, args=taskParas[i]))

    pool.close()
    pool.join()

    # for res in result:
    #     print('***:', res.get())  # get()?

    print('All end--')



def MainTask(npy_path,imgPath,saveRoot,slice_index,left_point,
             imgOrigin,spacing,refSize,block_size,
             ch,save_name_format):
    uz_path = os.path.join(saveRoot, save_name_format.format(slice_index + 1) + "_uz.mha")
    lz_path = os.path.join(saveRoot, save_name_format.format(slice_index + 1) + "_lz.mha")

    us_path = os.path.join(saveRoot, save_name_format.format(slice_index + 1) + "_us.mha")
    ls_path = os.path.join(saveRoot, save_name_format.format(slice_index + 1) + "_ls.mha")
    if os.path.exists(uz_path) and os.path.exists(ls_path):
        print(f"{save_name_format.format(slice_index + 1)} exists")
        return
    # if os.path.exists(os.path.join(saveRoot,
    #    "1293_NGGMDNR_1_{:03d}_561nm_10X_uz.mha".format(slice_index + 1))) and os.path.exists(os.path.join(saveRoot,
    #   "1293_NGGMDNR_1_{:03d}_561nm_10X_lz.mha".format(slice_index + 1))):
    #     return None
    rate = 4.0
    print(f"{npy_path} imgPath is {imgPath}, slice_index + 1 is {slice_index + 1}")
    if not os.path.exists(npy_path):
        print(f"{slice_index + 1} is not exists")
        OriginIndex = slice_index
        # InitCreatSurface(imgPath, saveRoot, slice_index + 1,imgOrigin,
        #                  refSize,left_point,ch,save_name_format)
        return
    data = np.load(npy_path)
    data = np.nan_to_num(data, nan=0.0)
    print(f"npy shape is {data.shape}")


    img = sitk.ReadImage(imgPath)

    img.SetOrigin(imgOrigin)
    #  spacing ?.0 ?sliceimage consistency
    img.SetSpacing(spacing)
    img_size = img.GetSize()
    img = sitk.Resample(img, [refSize[0], refSize[1], img_size[2]],
                           sitk.Transform(), sitk.sitkLinear, left_point, spacing)
    #  
    img.SetOrigin([0,0,0])
    img.SetSpacing([1,1,1])

    # todo
    gap = 38
    print(f"gap is {gap}")
    ls_ind = img_size[2] - gap
    end2 = img_size[2] - gap - 100
    # roi = [[first - interval, first], [second - interval, second]]

    # data = data[i:i+2,j:j+2,:]
    # data = np.load(os.path.join(r"D:\USERS\yq\code\cal_overlap\Refine\th2",'tf_' + str(120) + '_pars.npy'))
    # data = refine_npy.copy()
    shape = data.shape

    z = data[:, :, 2]  # z array needs to be 2-D
    rate_x = refSize[0] / (block_size * shape[0])
    rate_y = refSize[1] / (block_size * shape[1])
    x = np.arange(0, block_size * shape[0], block_size)  # the grid is an outer product
    y = np.arange(0, block_size * shape[1], block_size)  # of x and y arrays
    _, _, znew = CreateTransform(x, y, z, refSize)
    # _, _, znew = CreateTransform_Test(x, y, z, refSize)
    znew = end2 + znew / rate
    # znew = end2

    affine_t = [0, 1,
                1, 0]
    umap_z = sitk.GetImageFromArray(znew)
    umap_z = Flip(umap_z, affine_t)
    umap_z = sitk.Cast(umap_z, sitk.sitkFloat32)

    z = data[:, :, 0]
    _, _, x_trans = CreateTransform(x, y, z, refSize)
    # _, _, x_trans = CreateTransform_Test(x, y, z, refSize)
    x_trans = x_trans / rate
    umap_x = sitk.GetImageFromArray(x_trans)

    z = data[:, :, 1]
    _, _, y_trans = CreateTransform(x, y, z, refSize)
    # _, _, y_trans = CreateTransform_Test(x, y, z, refSize)
    y_trans = y_trans / rate
    umap_y = sitk.GetImageFromArray(y_trans)  # tips  ?
    umap_y = sitk.Cast(umap_y, sitk.sitkFloat32)

    umap_x = sitk.Cast(umap_x, sitk.sitkFloat32)

    umap_x = Flip(umap_x, affine_t)
    umap_y = Flip(umap_y, affine_t)
    uz = sitk.Compose(umap_x, umap_y, umap_z)

    lmap_x = sitk.Image(refSize, sitk.sitkFloat32)
    lmap_y = sitk.Image(refSize, sitk.sitkFloat32)
    lmap_z = sitk.Image(refSize, sitk.sitkFloat32) + ls_ind - 1
    lz = sitk.Compose(lmap_x, lmap_y, lmap_z)

    surfaces = copy_extract_surface(img, uz, lz)


    sitk.WriteImage(uz, uz_path)
    sitk.WriteImage(lz, lz_path)

    us = surfaces[:, :, 0]
    ls = surfaces[:, :, 1]
    sitk.WriteImage(us, us_path)
    sitk.WriteImage(ls, ls_path)

    # sitk.WriteImage(img[:, :, 75], os.path.join(saveRoot, "{}_std75.tif".format(slice_index + 1)))
    # sitk.WriteImage(img[:, :, 175], os.path.join(saveRoot, "{}_std175.tif".format(slice_index + 1)))
    print(f"img path is {imgPath}")
if __name__ == '__main__':
    # todo apply moved
    # for prevIndex in range(30,36):
    #     txtPath = r"D:\USERS\yq\code\cal_overlap\Refine\th2_0511\tf_{}_pars.txt".format(prevIndex)
    #     tempRoot = fr"D:\USERS\yq\code\cal_overlap\Refine\th2_0511\{prevIndex}_{prevIndex + 1}"
    #     ReadNpy(txtPath, tempRoot)


    multiCreateSurface()

    # # 88 single
    # index = 87
    # saveRoot = r"E:\wm1293\Reconstruction\Temp"
    # imgFormat = r"E:\wm1293\Reconstruction\SliceImage\4.0\1293_NGGMDNR_1_{:03d}_561nm_10X.tif"
    # # get visor point boundary
    # visorPath = r"E:\wm1293\20221024data.visor"
    # leftList, rightList = GetOffset(visorPath)
    # leftList = np.array(leftList)
    # rightList = np.array(rightList)
    #
    # spacing = [4, 4, 4]
    # lefttop = leftList.min(axis=0)
    # rightbottom = rightList.max(axis=0)
    # lefttop = [lefttop[0], lefttop[1], 0]
    # refSize = [(rightbottom[0] - lefttop[0]) // spacing[0], (rightbottom[1] - lefttop[1]) // spacing[1]]
    # refSize = [int(i) for i in refSize]
    # imgOrigin = leftList[index - 1]
    # InitCreatSurface(imgFormat.format(index), saveRoot, index, imgOrigin, refSize, lefttop)
