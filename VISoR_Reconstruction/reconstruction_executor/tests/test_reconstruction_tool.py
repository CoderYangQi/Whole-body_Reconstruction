import unittest

from VISoR_Reconstruction.reconstruction_executor.executor import main
from VISoR_Reconstruction.reconstruction_executor.generator import gen_brain_reconstruction_pipeline
from VISoR_Reconstruction.reconstruction_executor.roi_reconstruction_generator import gen_roi_reconstruction_pipeline
from VISoR_Brain.format.raw_data import RawData
from VISoR_Brain.misc import *


class ReconstructionToolTestCase(unittest.TestCase):
    def test_executor(self):
        #input_file = open(os.path.join(os.path.dirname(__file__), 'test_main.json'))
        input_file = open('F:/TEST_DATA/Mouse_Brain/20180914_ZMN_WH_438_1_1/Reconstruction/ReconstructionInput.json')
        input_file = input_file.read()
        main(input_file)

    def test_roi_reconstruction_generator(self):
        from VISoR_Brain.format.visor_data import VISoRData
        dataset = VISoRData('D:/Hao/Data/converted/cfos-C2_2652/Reconstruction.visor')
        param = {'roi': [[1000, 1000, 1000], [2000, 2000, 2000]]}
        doc = gen_roi_reconstruction_pipeline(dataset, **param)
        with open('F:/chaoyu/test/t.txt', 'w') as f:
            f.write(doc)


    def test_generator(self):
        raw_data_s = ["Z:/VISoR12_data/VISoR12/Macaque/RM005/PBM_RM_111_120/20180731_ZMN_PBM_CL-SC2_RM005_120/488nm_10X/120_120/PBM_CL-SC2_RM005_120_120.flsm",
"Z:/VISoR12_data/VISoR12/Macaque/RM005/PBM_RM_111_120/20181018_ZMN_PBM_CLA_RM005_116-/Data/488nm_10X/116_116/PBM_CLA_RM005_116_116.flsm",
"Z:/VISoR12_data/VISoR12/Macaque/RM005/PBM_RM_111_120/20181018_ZMN_PBM_CLA_RM005_117/Data/488nm_10X/117_117/PBM_CLA_RM005_117_117_1.flsm",
"Z:/VISoR12_data/VISoR12/Macaque/RM005/PBM_RM_111_120/20181023_ZMN_PBM_CLA_RM005_118/Data/488nm_10X/118_118/PBM_CLA_RM005_118_118.flsm",
"Z:/VISoR12_data/VISoR12/Macaque/RM005/PBM_RM_111_120/20181023_ZMN_PBM_CLA_RM005_119-/Data/488nm_10X/119_119/PBM_CLA_RM005_119_119_1.flsm",
"U:/Macaque/PBM_RM005_121_128/20181025_ZMN_PBM_CLA_RM005_121/Data/488nm_10X/121_121/PBM_CLA_RM005_121_121_1.flsm",
"U:/Macaque/PBM_RM005_121_128/20181025_ZMN_PBM_CLA_RM005_122/Data/488nm_10X/122_122/PBM_CLA_RM005_122_122.flsm",
"U:/Macaque/PBM_RM005_121_128/20181026_ZMN_PBM_CLA_RM005_123/Data/488nm_10X/123_123/PBM_CLA_RM005_123_123_1.flsm",
"U:/Macaque/PBM_RM005_121_128/20181025_ZMN_PBM_CLA_RM005_124/Data/488nm_10X/124_124/PBM_CLA_RM005_124_124_1.flsm",
"U:/Macaque/PBM_RM005_121_128/20181030_ZMN_PBM_CLA_RM005_125/Data/488nm_10X/125_125/PBM_CLA_RM005_125_125_1.flsm",
"U:/Macaque/PBM_RM005_121_128/20181030_ZMN_PBM_CLA_RM005_126/Data/488nm_10X/126_126/PBM_CLA_RM005_126_126.flsm",
"U:/Macaque/PBM_RM005_121_128/20181101_ZMN_PBM_CLA_RM005_127/Data/488nm_10X/127_127/PBM_CLA_RM005_127_127_1.flsm",
"U:/Macaque/PBM_RM005_121_128/20181101_ZMN_PBM_CLA_RM005_128/Data/488nm_10X/128_128/PBM_CLA_RM005_128_128.flsm",
"T:/Macaque/20181116_ZMN_PBM_CLA_RM005_129/Data/488nm_10X/129_129/PBM_CLA_RM005_129_129_1.flsm",
"D:/VISoR12/Macaque/PBM_RM005/20180914_ZMN_RM005_110_1_110/Data/488nm_10X/RM005_110_1_110.flsm",
"Z:/VISoR12_data/VISoR12/Macaque/RM005/PBM_RM_111_120/20180920_ZMN_RM005_110_1_111/Data/488nm_10X/RM005_110_1_111.flsm",
"Z:/VISoR12_data/VISoR12/Macaque/RM005/PBM_RM_111_120/20180920_ZMN_RM005_112_1_112/Data/488nm_10X/RM005_112_1_112.flsm",
"Z:/VISoR12_data/VISoR12/Macaque/RM005/PBM_RM_111_120/20180921_ZMN_RM005_113_1_113/Data/488nm_10X/RM005_113_1_113.flsm",
"Z:/VISoR12_data/VISoR12/Macaque/RM005/PBM_RM_111_120/20180921_ZMN_RM005_114_1_114/Data/488nm_10X/RM005_114_1_114_3.flsm",
"Z:/VISoR12_data/VISoR12/Macaque/RM005/PBM_RM_111_120/20180921_ZMN_RM005_115_1_115/Data/488nm_10X/RM005_115_1_115.flsm"]
        '''
        for root, dirs, files in os.walk('F:/TEST_DATA/Mouse_Brain/20180914_ZMN_WH_438_1_1'):
            for file in files:
                if file.split('.')[-1] == 'flsm':
                    try:
                        r = RawData(os.path.join(root, file))
                        #if not 20 <= int(r.z_index) <= 25:
                        #    continue
                    except Exception as e:
                        print(e)
                        continue
                    print(file)
                    raw_data_list.append(r)
        '''
        raw_data_list = []
        for r in raw_data_s:
            r = RawData(r)
            r.caption = 'RM005'
            raw_data_list.append(r)
        s = gen_brain_reconstruction_pipeline(raw_data_list, reference_channel='488', preset='macaque_fast',
                                              output_path='D:/Fang/PBM_RM005')
        f = open('F:/chaoyu/test/t.txt', 'w')
        f.write(s)
        main(s)


