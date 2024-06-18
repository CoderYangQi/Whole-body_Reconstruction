import unittest
from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
from VISoR_Brain.utils.elastix_files import *
from VISoR_Reconstruction.reconstruction.brain_reconstruct_methods.common import fill_outside
from common_script.common0424 import *
from ReconstructionScripts.Step1 import GetOffset

def RefineImg(imgPath, refSize, imgOrigin,spacing, lefttop):
    # imgPath = imgFormat.format(index)
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
    return img

class TestAddFunction(unittest.TestCase):
    def test_print(self):
        print("test")

    def test_Img111_112(self):
        roi = [[3510,4624], [4248,5076]]
        # 计算ROI覆盖的起始和结束块索引
        block_size = 250
        # for i in range(row): #     for j in range(col):
        start_row = roi[0][0] // block_size
        end_row = (roi[1][0] - 1) // block_size
        start_col = roi[0][1] // block_size
        end_col = (roi[1][1] - 1) // block_size

        # init parameters
        saveRoot = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\TestTemp\th2_111_112"
        imgFormat = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\SliceImage\4.0\2_1_1_{:03d}_561nm_10X.tif"
        npyFormat = r"D:\USERS\yq\code\cal_overlap\Refine\th2_0511\0511_refine_{}_pars.npy"
        # get flsm data
        # from ReconstructionScripts.Step1 import GetOffset
        visorPath = r"D:\USERS\yq\TH2_Reconstruction\delete145New.visor"
        leftList, rightList = GetOffset(visorPath)
        leftList = np.array(leftList)
        rightList = np.array(rightList)

        spacing = [4, 4, 4]
        lefttop = leftList.min(axis=0)
        rightbottom = rightList.max(axis=0)
        lefttop = [lefttop[0], lefttop[1], 0]
        refSize = [(rightbottom[0] - lefttop[0]) // spacing[0], (rightbottom[1] - lefttop[1]) // spacing[1]]
        refSize = [int(i) for i in refSize]
        # InitCreatSurface(imgFormat.format(index),visorPath,saveRoot,index, OriginIndex=index-1)


        for sliceIndex in range(111,113):
            imgPath = imgFormat.format(sliceIndex)
            imgOrigin = leftList[sliceIndex - 1]
            imgOrigin = [imgOrigin[0], imgOrigin[1], 0]
            spacing = [4, 4, 4]
            refineImg = RefineImg(imgPath, refSize, imgOrigin, spacing, lefttop)
            temp = refineImg[roi[0][0]:roi[1][0],roi[0][1]:roi[1][1],:]
            sitk.WriteImage(temp,
                            os.path.join(saveRoot,"{:03d}_roi.tif".format(sliceIndex)))



