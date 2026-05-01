#from VISoR_Brain.positioning.visor_sample import *
import unittest
import SimpleITK as sitk
import numpy as np
from lib import flsmio

sample_file = 'D:/chaoyu/Test/DLF_VIRAL_TRACING_WUHAN-2_2/24.tar'


class VISoRSampleTestCase(unittest.TestCase):
    def setUp(self):
        #self.sample = VISoRSample()
        #self.sample.load(sample_file)
        #self.sample.load_columns()
        pass

    def test_get_sample_image(self):
        img = self.sample.get_sample_image(self.sample.sphere, 4, source='thumbnail')
        sitk.WriteImage(img, 'D:/chaoyu/Test/DLF_VIRAL_TRACING_WUHAN-2_2/24.mha')
        img = sitk.GetArrayFromImage(img)
        average = np.average(img)
        self.assertAlmostEqual(average, 141.983, delta=0.1)

    def test_raw_data_position(self):
        reader = flsmio.FlsmReader('V:/VISoR12/Human/20200530_SY_ANATOMY_SECOND_1_1/Data/488nm_10X/ANATOMY_SECOND_1_1.flsm')
        all_size = reader.size()
        n_stacks = int(all_size[0])
        n_images = int(all_size[1])
        for i in range(n_images):
            image = reader.thumbnail(8, i)
            if image is None:
                continue
            pos = image.position()
            print(pos[0])

