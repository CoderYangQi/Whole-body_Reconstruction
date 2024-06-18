import json
import unittest
from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
from VISoR_Brain.utils.elastix_files import *
from VISoR_Reconstruction.reconstruction.brain_reconstruct_methods.common import fill_outside
from common_script.common0424 import *
import multiprocessing
import time, gc

def run_multiprocess(numsThread, taskParas):
    # todo use multiprocess
    pool = multiprocessing.Pool(numsThread)
    result = []
    for i in range(len(taskParas)):
        msg = 'hello %s' % i
        result.append(pool.apply_async(func=Task, args=taskParas[i]))

    pool.close()
    pool.join()
def yq_generate_brain_image(brain: VISoRBrain, img, slice_index, input_pixel_size, output_pixel_size, name_format, n_start,
                         roi=None, slice_origin=None, bit_downsample=True):
    slice_origin = [0,0,0]
    # if slice_origin is None:
    #     slice_origin = brain.slices[slice_index].sphere[0]
    #     # # todo 可能是数据有问题
    #     # slice_origin[2] = 0
    img.SetOrigin([0,0,0])
    img.SetSpacing([input_pixel_size, input_pixel_size, input_pixel_size])
    if roi is None:
        roi = brain.slice_spheres[slice_index]
    size = [int((roi[1][j] - roi[0][j]) / output_pixel_size)
            for j in range(3)]
    print(size)
    print(f"roi is {roi}")
    tempTransform = brain.transform(slice_index)
    res = sitk.Resample(img, size, brain.transform(slice_index), sitk.sitkLinear, roi[0],
                        [output_pixel_size, output_pixel_size, output_pixel_size])




    res.SetSpacing([j / 1000 for j in res.GetSpacing()])
    paths = [name_format.format(n_start + j) for j in range(size[2])]
    if not os.path.exists(os.path.dirname(paths[0])):
        os.makedirs(os.path.dirname(paths[0]))
    for i in range(size[2]):
        m = sitk.GetArrayFromImage(res[:, :, i])
        if bit_downsample:
            m = np.left_shift(np.right_shift((m + 8), 4), 4)
        tifffile.imwrite(paths[i], m, compress=1)
    file_list = paths.__str__()[2:-2].replace('\', \'', '\n')
    return file_list
def main():
    # todo 对img做位置的初始化
    saveRoot = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\TestTemp\th2_30_36"
    # imgFormat = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\SliceImage\4.0\2_1_1_{:03d}_561nm_10X.tif"
    imgFormat = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\SliceImage\4.0\2_1_1_{:03d}_561nm_10X.tif"
    # get flsm data
    from ReconstructionScripts.Step1 import GetOffset
    visorPath = r"D:\USERS\yq\TH2_Reconstruction\delete145New.visor"
    leftList, rightList = GetOffset(visorPath)
    leftList = np.array(leftList)
    rightList = np.array(rightList)

    spacing = [4, 4, 4]
    lefttop = leftList.min(axis=0)
    rightbottom = rightList.max(axis=0)
    lefttop = [lefttop[0], lefttop[1], 0]
    refSize = [(rightbottom[0] - lefttop[0]) // spacing[0], (rightbottom[1] - lefttop[1]) // spacing[1] + 750]
    refSize = [int(i) for i in refSize]
    # brainPath = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\BrainTransform\visor_brain.txt"
    brainPath = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\BrainTransform\visor_brain.txt"
    name_format = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\BrainImage\4.0\Z{:05d}_C1.tif"
    taskChunks = []
    for sliceIndex in range(1,185):
        # sliceIndex = 30  # index 从 1 开始算
        indexOrigin = sliceIndex - 1
        imgOrigin = leftList[indexOrigin]
        taskChunks.append((sliceIndex, imgOrigin, imgFormat, refSize, spacing, lefttop, brainPath,name_format))

    num_threads = 5  # 设置线程数量
    run_multiprocess(num_threads, taskChunks)

def Task(sliceIndex, imgOrigin, imgFormat, refSize, spacing, lefttop,brainPath,name_format):
    # 初始化 image
    brain = VISoRBrain()
    brain.load(brainPath)
    imgOrigin = [imgOrigin[0], imgOrigin[1], 0]
    imgPath = imgFormat.format(sliceIndex)
    img = sitk.ReadImage(imgPath)
    nextSize = img.GetSize()
    # img_size = [nextSize[0], nextSize[1]]
    # todo 对图像进行 Resample 和之前的计算粗校准面的坐标一直
    img.SetOrigin(imgOrigin)
    img.SetSpacing(spacing)
    img = sitk.Resample(img, [refSize[0], refSize[1], nextSize[2]], sitk.Transform(), sitk.sitkLinear, lefttop,
                        [4, 4, 4])
    img.SetOrigin([0, 0, 0])
    img.SetSpacing([1, 1, 1])

    input_pixel_size = 4
    output_pixel_size = 4
    n_start = (sliceIndex - 1) * 100


    yq_generate_brain_image(brain, img, sliceIndex, input_pixel_size, output_pixel_size, name_format, n_start,
                            roi=None, slice_origin=None, bit_downsample=True)


if __name__ == '__main__':
    main()