from VISoR_Reconstruction.reconstruction.sample_reconstruct import *
import unittest


class SampleReconstructTestCase(unittest.TestCase):

    def test_reconstruct_sample(self):
        #raw0 = RawData('R:/VISoR12/Macaque/PBM_RM006/20190409_SY_PBM_HB_RM006_122/Data/488nm_10X/122_122/PBM_HB_RM006_122_122_1.flsm')
        raw = RawData('R:/VISoR12/Macaque/PBM_RM006/20190409_SY_PBM_HB_RM006_122/Data/561nm_10X/122_122/PBM_HB_RM006_122_122.flsm')
        #ref = reconstruct_sample(raw0, {'stitch': 'elastix_align2'})
        #ref.save('F:/chaoyu/test/122_488.txt')
        ref = VISoRSample('F:/chaoyu/test/122_488.txt')
        sl = reconstruct_sample(raw, {'align_channels': 'channel_elastix_align'}, ref)
        sl.save('F:/chaoyu/test/122_552.txt')
        #image = reconstruct_image(sl, 4, ref.sphere, source='thumbnail')
        #sitk.WriteImage(image, os.path.join(TEST_DATA_DIR, 'visor12_data/PBM_AAV-NT640_G07/552/PBM_AAV-NT640_G07_1_006.tif'))
