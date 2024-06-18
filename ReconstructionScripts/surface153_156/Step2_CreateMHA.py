'''
@ yangqi
def test_ReadNpy(self): 应用translate prameters 查看计算结果的 moved 效果

'''

import json
import unittest
from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
from VISoR_Brain.utils.elastix_files import *
from VISoR_Reconstruction.reconstruction.brain_reconstruct_methods.common import fill_outside
from common_script.common0424 import *

def ReadOffsetTxt(txtPath = r"D:\USERS\yq\code\cal_overlap\Refine\th2_33\tf_33_pars.txt"):
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
def Flip(img,affine_t):


    af = sitk.AffineTransform(2)
    af.SetMatrix(affine_t)
    size = img.GetSize()
    size = [size[1], size[0]]
    sitk_image = sitk.Resample(img, size, af)
    # write_ome_tiff(sitk_image,'temp.tif')
    return sitk_image

# 同 cal overlap
def ReadNpy(txtPath,tempName):
    import re

    # 初始化一个空字典来存储坐标和对应的偏移量
    offsets = {}

    # 假设文本文件的内容已经以字符串形式给出或已经保存在一个文件中
    # txtPath = r"D:\USERS\yq\code\cal_overlap\Refine\th2_0511\tf_33_pars.txt"
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
import multiprocessing
import time, gc

def run_multiprocess(numsThread, taskParas):
    # todo use multiprocess
    pool = multiprocessing.Pool(numsThread)
    result = []
    for i in range(len(taskParas)):
        msg = 'hello %s' % i
        result.append(pool.apply_async(func=MainTask, args=taskParas[i]))

    pool.close()
    pool.join()

    # for res in result:
    #     print('***:', res.get())  # get()函数得出每个返回结果的值

    print('All end--')
def multiCreateSurface():
    saveRoot = r"Z:\Data\E\E-123\Reconstruction\saveTemp\th0630_153_155"
    imgFormat = r"Z:\Data\E\E-123\Reconstruction\SliceImage\{}_img.tif"
    npyFormat = r"Z:\Data\E\E-123\Reconstruction\saveTemp\th0630_153_155\0526_refine_{}_pars.npy"
    # get flsm data
    # from ReconstructionScripts.Step1 import GetOffset

    refSize = [8375, 4500]

    ###

    spacing = [1,1,1]
    block_size = 250
    sub_block = 125
    interval = 40
    end2 = 270
    end2 = 75
    # todo start
    # 如果从 0 开始计算 那么就是从 sliceIndex + 1 开始做create
    taskChunk = []
    lefttop = [0,0,0]
    for slice_index in range(153,156):
        originIndex = slice_index
        npy_path = npyFormat.format(slice_index)
        imgPath = imgFormat.format(slice_index + 1)
        imgOrigin = [0,0,0]
        tempChunk = (npy_path,imgPath,saveRoot,slice_index,
             imgOrigin,spacing,refSize,lefttop,block_size,end2)
        taskChunk.append(tempChunk)

    num_threads = 4  # 设置线程数量
    run_multiprocess(num_threads, taskChunk)


# todo 只制作第一个数据
from ReconstructionScripts.Step1 import GetOffset
def InitCreatSurface(imgPath,saveRoot,index,imgOrigin, refSize,lefttop):
    # 初始化 image
    imgOrigin = [imgOrigin[0], imgOrigin[1], 0]
    spacing = [1, 1, 1]

    # imgPath = imgFormat.format(index)
    img = sitk.ReadImage(imgPath)
    nextSize = img.GetSize()
    # img_size = [nextSize[0], nextSize[1]]
    # todo 对图像进行 Resample 和之前的计算粗校准面的坐标一直
    img.SetOrigin(imgOrigin)
    img.SetSpacing(spacing)
    img = sitk.Resample(img, [refSize[0], refSize[1], nextSize[2]], sitk.Transform(), sitk.sitkLinear, lefttop,
                        spacing)
    img.SetOrigin([0, 0, 0])
    img.SetSpacing([1, 1, 1])



    ### todo surface Displace
    umap_x = sitk.Image(refSize, sitk.sitkFloat32)
    umap_y = sitk.Image(refSize, sitk.sitkFloat32)
    umap_z = sitk.Image(refSize, sitk.sitkFloat32) + 75
    uz = sitk.Compose(umap_x, umap_y, umap_z)

    lmap_x = sitk.Image(refSize, sitk.sitkFloat32)
    lmap_y = sitk.Image(refSize, sitk.sitkFloat32)
    lmap_z = sitk.Image(refSize, sitk.sitkFloat32) + 175 - 1
    lz = sitk.Compose(lmap_x, lmap_y, lmap_z)

    surfaces = copy_extract_surface(img, uz, lz)
    sitk.WriteImage(uz, os.path.join(saveRoot, "2_1_1_{:03d}_561nm_10X_uz.mha".format(index)))
    sitk.WriteImage(lz, os.path.join(saveRoot, "2_1_1_{:03d}_561nm_10X_lz.mha".format(index)))

    sitk.WriteImage(surfaces[:, :, 0], os.path.join(saveRoot, "2_1_1_{:03d}_561nm_10X_us.mha".format(index)))
    sitk.WriteImage(surfaces[:, :, 1], os.path.join(saveRoot, "2_1_1_{:03d}_561nm_10X_ls.mha".format(index)))

def MainTask(npy_path,imgPath,saveRoot,slice_index,
             imgOrigin,spacing,refSize,lefttop,block_size,end2):
    rate = 1.0
    print(f"{npy_path} imgPath is {imgPath}, slice_index + 1 is {slice_index + 1}")
    if not os.path.exists(npy_path):
        print(f"{slice_index + 1} is not exists")
        OriginIndex = slice_index
        InitCreatSurface(imgPath, saveRoot, slice_index + 1,imgOrigin,refSize,lefttop)
        return
    data = np.load(npy_path)
    print(f"npy shape is {data.shape}")


    img = sitk.ReadImage(imgPath)

    #  end
    # 复原 origin
    # img = sitk.GetArrayFromImage(img)
    # img = sitk.GetImageFromArray(img)
    img.SetOrigin([0, 0, 0])
    img.SetSpacing([1, 1, 1])

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
    znew = end2 + znew / rate
    # znew = end2

    affine_t = [0, 1,
                1, 0]
    umap_z = sitk.GetImageFromArray(znew)
    umap_z = Flip(umap_z, affine_t)
    umap_z = sitk.Cast(umap_z, sitk.sitkFloat32)

    z = data[:, :, 0] / rate
    _, _, x_trans = CreateTransform(x, y, z, refSize)
    umap_x = sitk.GetImageFromArray(x_trans)

    z = data[:, :, 1] / rate
    _, _, y_trans = CreateTransform(x, y, z, refSize)
    umap_y = sitk.GetImageFromArray(y_trans)  # tips  可能需要取反
    umap_y = sitk.Cast(umap_y, sitk.sitkFloat32)

    umap_x = sitk.Cast(umap_x, sitk.sitkFloat32)

    umap_x = Flip(umap_x, affine_t)
    umap_y = Flip(umap_y, affine_t)
    uz = sitk.Compose(umap_x, umap_y, umap_z)

    lmap_x = sitk.Image(refSize, sitk.sitkFloat32)
    lmap_y = sitk.Image(refSize, sitk.sitkFloat32)
    if slice_index + 1 == 87:
        lmap_z = sitk.Image(refSize, sitk.sitkFloat32) + 175 - 1
    else:
        lmap_z = sitk.Image(refSize, sitk.sitkFloat32) + 175 - 1
    lz = sitk.Compose(lmap_x, lmap_y, lmap_z)

    surfaces = copy_extract_surface(img, uz, lz)
    sitk.WriteImage(uz, os.path.join(saveRoot, "2_1_1_{:03d}_561nm_10X_uz.mha".format(slice_index + 1)))
    sitk.WriteImage(lz, os.path.join(saveRoot, "2_1_1_{:03d}_561nm_10X_lz.mha".format(slice_index + 1)))

    us = surfaces[:, :, 0]
    ls = surfaces[:, :, 1]
    sitk.WriteImage(us, os.path.join(saveRoot, "2_1_1_{:03d}_561nm_10X_us.mha".format(slice_index + 1)))
    sitk.WriteImage(ls, os.path.join(saveRoot, "2_1_1_{:03d}_561nm_10X_ls.mha".format(slice_index + 1)))

    sitk.WriteImage(img[:, :, 75], os.path.join(saveRoot, "{}_std75.tif".format(slice_index + 1)))
    sitk.WriteImage(img[:, :, 175], os.path.join(saveRoot, "{}_std175.tif".format(slice_index + 1)))
    print(f"img path is {imgPath}")
if __name__ == '__main__':
    # todo apply moved
    # for prevIndex in range(30,36):
    #     txtPath = r"D:\USERS\yq\code\cal_overlap\Refine\th2_0511\tf_{}_pars.txt".format(prevIndex)
    #     tempRoot = fr"D:\USERS\yq\code\cal_overlap\Refine\th2_0511\{prevIndex}_{prevIndex + 1}"
    #     ReadNpy(txtPath, tempRoot)


    multiCreateSurface()

    # # 88 single
    index = 153
    saveRoot = r"Z:\Data\E\E-123\Reconstruction\saveTemp\th0630_153_155"
    imgFormat = r"Z:\Data\E\E-123\Reconstruction\SliceImage\{}_img.tif"
    npyFormat = r"Z:\Data\E\E-123\Reconstruction\saveTemp\th0630_153_155\0526_refine_{}_pars.npy"
    # get flsm data
    # from ReconstructionScripts.Step1 import GetOffset
    lefttop = [0,0,0]
    refSize = [8375, 4500]


    InitCreatSurface(imgFormat.format(index), saveRoot, index, lefttop, refSize, lefttop)