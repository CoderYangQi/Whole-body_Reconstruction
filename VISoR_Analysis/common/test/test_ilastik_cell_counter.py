from VISoR_Analysis.common.ilastik_cell_counter import *
import unittest

class IlastikCellCounterTestCase(unittest.TestCase):
    def test_run_ilastik(self):
        run_ilastik('F:/chaoyu/test/thy1/cells.ilp', 'F:/chaoyu/test/thy1/sample.tif', 'F:/chaoyu/test/thy1')
