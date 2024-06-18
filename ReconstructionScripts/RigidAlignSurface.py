'''
#@ yangqi
1. step1()  使用sitk 得到 75：175的数据

2. step2()  使用 sitk 计算translate

'''

import argparse
import json
import re
import unittest
import os
import SimpleITK as sitk
from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
from common_script.common0424 import *
def get_files(image_root, extend_name):
    image_files_path = os.listdir(image_root)
    path_temp = []
    for i in image_files_path:
        fname, ext = os.path.splitext(i)
        if ext == extend_name:
            i = os.path.join(image_root, i)
            path_temp.append(i)
        pass
    image_files_path = path_temp
    return  image_files_path

import multiprocessing
import time, gc

def run_multiprocess(numsThread, taskParas):
    # todo use multiprocess
    pool = multiprocessing.Pool(numsThread)
    result = []
    for i in range(len(taskParas)):
        msg = 'hello %s' % i
        # print(taskParas[i])
        result.append(pool.apply_async(func=Task, args=taskParas[i]))

    pool.close()
    pool.join()

    # for res in result:
    #     print('***:', res.get())  # get()函数得出每个返回结果的值

    print('All end--')

def CalChannelTranslate(prev_surface, next_surface):
    def PreProcess(img):
        img = sitk.Cast(img, sitk.sitkFloat32)
        refineImg = sitk.Clamp((sitk.Log(sitk.Cast(img, sitk.sitkFloat32)) - 4.6) * 39.4, sitk.sitkUInt8, 0, 255)
        return refineImg
    translateDict = {}
    tempName = os.path.join(PARAMETER_DIR, 'yq_channel_align_surface_2D.txt')
    # 选取 index 为 33 的数据进行测试
    # prev_surface = sitk.ReadImage(prev_surface_path)
    # next_surface = sitk.ReadImage(next_surface_path)
    prev_surface = PreProcess(prev_surface)
    next_surface = PreProcess(next_surface)
    prev_size = prev_surface.GetSize()
    next_size = next_surface.GetSize()
    ref_scale = 1
    outside_brightness = 2
    # next_surface = ResizeImg(next_surface, next_size, ref_scale)
    prev_surface = fill_outside(prev_surface, outside_brightness)
    next_surface = fill_outside(next_surface, outside_brightness)
    # 此处 按照 next 为 fixed ； prev 为 moving； 因为需要计算 next的上表面偏移量

    tp_ = translate_get_align_transform(next_surface, prev_surface,
                                        [os.path.join(PARAMETER_DIR,
                                                      'yq_align_surface_2D.txt')])

    return tp_

def ApplyTranslate(img,offset):
    translate = [offset[0],offset[1],0]

    # 创建平移变换
    translation = sitk.TranslationTransform(3, translate)
    # 重采样过滤器
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)  # 设置参考图像
    resampler.SetInterpolator(sitk.sitkLinear)  # 设置插值方法
    resampler.SetTransform(translation)  # 设置应用的平移变换

    # 用平移变换重采样图像
    resampled_image = resampler.Execute(img)

    # 保存重采样后的图像

    return img

# 封装 step1 将执行代码到下面task中
def Task(refImgPath,imgPath,saveImgPath,saveRoot,sliceIndex):



    refImg = sitk.ReadImage(refImgPath)
    img = sitk.ReadImage(imgPath)

    refSize = refImg.GetSize()
    img = sitk.Resample(img, refImg)

    maxRefImg = sitk.MaximumProjection(refImg[:, :, 700:720], projectionDimension=2)[:, :, 0]
    maxImg = sitk.MaximumProjection(img[:, :, 280:300], projectionDimension=2)[:, :, 0]
    tp_ = CalChannelTranslate(maxRefImg, maxImg)
    # offsetDict[sliceIndex] = tp_

    img = ApplyTranslate(img, tp_)
    sitk.WriteImage(img[:, :, 300], os.path.join(saveRoot, "refineImg_{:03d}.tif".format(sliceIndex)))
    sitk.WriteImage(refImg[:, :, 720], os.path.join(saveRoot, "refImg_{:03d}.tif".format(sliceIndex)))
    with open(os.path.join(saveRoot, f"{sliceIndex}_tp.txt"), 'w') as file:
        # 写入数据，每个值占一行
        file.write(f"{tp_}\n")
    file.close()
    write_ome_tiff(img, saveImgPath)



# todo get maxprojection from image
def Step1():
    # root = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\SliceImage\4.0"
    imgFormat = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\test_{}.tif"
    sliceIndex = 77
    refImgFormat = imgFormat.format(sliceIndex)
    ImgFormat = imgFormat.format(sliceIndex + 1)
    saveRoot = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction"
    taskChunk = []
    for sliceIndex in range(77,78):
        saveImgPath = os.path.join(saveRoot, "refine_{}.tif".format(sliceIndex))
        if os.path.exists(saveImgPath):
            continue
        offsetDict = {}
        refImgPath = refImgFormat.format(sliceIndex)
        imgPath = ImgFormat.format(sliceIndex)
        tempChunk = (refImgPath, imgPath, saveImgPath, saveRoot, sliceIndex)
        taskChunk.append(tempChunk)
    num_threads = 5  # 设置线程数量
    run_multiprocess(num_threads, taskChunk)

    return None


if __name__ == '__main__':
    Step1()