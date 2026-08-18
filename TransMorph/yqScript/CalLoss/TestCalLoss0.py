# calculate the transmorph disp loss
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import unittest,os
import SimpleITK as sitk

class TestSplitAndTortue(unittest.TestCase):
    def test_print(self):
        print("Testing calloss")
    def test_readImg(self):
        root = r"Z:\users\yq\MorphDatasets\model\TransMorph\0813\test_result"
        # imgFormat, savePath
        # imgFormat = os.path.join(root, ('y_{:04d}' + '.tif'))
        # savePath = os.path.join(root,"y_out.mha")
        # self.test_combineFun(imgFormat, savePath)

        imgFormat = os.path.join(root, ('x_def_{:04d}' + '.tif'))
        savePath_x_def = os.path.join(root, "x_def_out.mha")
        # savePath_y = os.path.join(root,"y_out.mha")
        img0 = sitk.ReadImage()