from VISoR_Brain.format.raw_data import *
from VISoR_Brain.format.visor_data import *
from VISoR_Brain.misc import *
import unittest


class RawDataTestCase(unittest.TestCase):
    def test_raw_data_v1(self):
        flsm_file = os.path.join(TEST_DATA_DIR, 'visor1_data/2017-01-19_09-47-31/2017-01-19_09-47-31.flsm')
        rawdata = RawData(flsm_file, os.path.join(ROOT_DIR, 'devices/visor1.txt'))

    def test_raw_data_v2(self):
        flsm_file = os.path.join(TEST_DATA_DIR, 'visor12_data/1_006/LC_PROJECTION_2473_1_006.flsm')
        rawdata = RawData(flsm_file, os.path.join(ROOT_DIR))

    def test_visor_data(self):
        visor_data = VISoRData('E:/FTP/USTC_SIST/Test/20181225_ZMN_424_NT640_STRAIN_1/Data.visor')
        print(visor_data.__dict__)
        visor_data.save('F:/chaoyu/test/test.visor')

    def test_visor_data2(self):
        rawdatalist = []
        for root, dirs, files in os.walk('F:/TEST_DATA/Mouse_Brain/20180914_ZMN_WH_438_1_1'):
            for file in files:
                if file.split('.')[-1] == 'flsm':
                    try:
                        r = RawData(os.path.join(root, file))
                    except Exception as e:
                        print(e.__traceback__)
                        print(e)
                        continue
                    print(file)
                    rawdatalist.append(r)
        visor_data = VISoRData(raw_data_list=rawdatalist)
        print(visor_data.__dict__)
        visor_data.save('F:/TEST_DATA/Mouse_Brain/20180914_ZMN_WH_438_1_1/Data.visor')

    def test_visor_data3(self):
        visor_data = VISoRData(r'X:\data\XC-beads\20220623_XC_TEST_BEAD_0/Reconstruction.visor')
        print(visor_data.slice_transform)

