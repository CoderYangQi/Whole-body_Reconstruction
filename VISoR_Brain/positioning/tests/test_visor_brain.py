import unittest, random, warnings
from VISoR_Brain.positioning.visor_brain import VISoRBrain

brain = VISoRBrain('F:/TEST_DATA/Mouse_Brain/20180914_ZMN_WH_438_1_1/Reconstruction/BrainTransform/visor_brain.txt')
class VISoRSampleTestCase(unittest.TestCase):

    def test_get_column_position(self):
        brain_pos = [4000, 4000, 5000]
        raw_pos = brain.get_column_position(brain_pos)
        print(raw_pos)

    def test_get_slice_position(self):
        brain_pos = [4000, 4000, 6001]
        slice_pos = brain.get_slice_position(brain_pos, 20)
        print(slice_pos)

    def test_get_brain_position_from_slice(self):
        ct = 0
        for i in brain.slices:
            sl = brain.slices[i]
            for j in range(10000):
                slice_pos = [random.random() * (sl.sphere[1][k] - sl.sphere[0][k]) + sl.sphere[0][k] for k in range(3)]
                with warnings.catch_warnings(record=True) as w:
                    brain_pos = brain.get_brain_position_from_slice(i, slice_pos)
                    if len(w) > 0:
                        ct += 1
        print(ct)

    def test_get_brain_position_from_column(self):
        brain_pos = brain.get_brain_position_from_column(22, 3, [1200, 400, 1500])
        print(brain_pos)
