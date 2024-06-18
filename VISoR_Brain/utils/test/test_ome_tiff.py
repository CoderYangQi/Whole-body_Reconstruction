from VISoR_Brain.utils.ome_tiff import *
import unittest, os
from VISoR_Brain.misc import *

class OMETiffTestCase(unittest.TestCase):
    def test_write_ome_tiff(self):
        image = sitk.ReadImage('F:/TEST_DATA/Mouse_Brain/20180914_ZMN_WH_438_1_1/Reconstruction/Slice/4/WH_438_1_1_013_488.tif')
        write_ome_tiff(image, 'E:/1.tif')