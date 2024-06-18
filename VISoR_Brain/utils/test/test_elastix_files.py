import os
import unittest
from VISoR_Brain.utils.elastix_files import *
from VISoR_Brain.misc import ROOT_DIR

class ElastixFilesTestCase(unittest.TestCase):
    def test_read_elastix_parameter(self):
        param = read_elastix_parameter('F:/chaoyu/test/TransformParameters.1.txt')
        print(param)

    def test_get_sitk_transform1(self):
        param = read_elastix_parameter('D:/Hao/reconstructed/cfos_counting/c-fos-FS3-577_488/res/par/b.i.25.0.txt')
        tf, im = get_sitk_transform(param)
        df = sitk.TransformToDisplacementField(tf, sitk.sitkVectorFloat64, im['Size'], im['Origin'], im["Spacing"], im['Direction'])
        jac = sitk.DisplacementFieldJacobianDeterminant(df)
        sitk.WriteImage(jac, 'F:/chaoyu/test/jac.mha')
        print(tf, im)
        print(tf.GetParameters())

    def test_get_sitk_transform2(self):
        param = sitk.ReadParameterFile('F:/chaoyu/test/thy1/438/TransformParameters.2.txt')
        tf, im = get_sitk_transform(param)
        df = sitk.TransformToDisplacementField(tf, sitk.sitkVectorFloat64, im['Size'], im['Origin'], im["Spacing"], im['Direction'])
        jac = sitk.DisplacementFieldJacobianDeterminant(df)
        sitk.WriteImage(jac, 'F:/chaoyu/test/jac.mha')
        print(tf, im)
        print(tf.GetParameters())

    def test_get_sitk_transform_from_file(self):
        tf, im = get_sitk_transform_from_file('F:/chaoyu/test/thy1/438/TransformParameters.2.txt')
        print(tf)
        df = sitk.TransformToDisplacementField(tf, sitk.sitkVectorFloat64, im['Size'], im['Origin'], im["Spacing"], im['Direction'])
        jac = sitk.DisplacementFieldJacobianDeterminant(df)
        sitk.WriteImage(jac, 'F:/chaoyu/test/jac.mha')

    def test_get_align_transform(self):
        prev_surface = sitk.ReadImage('F:/NewData/2016-10-03_Thy1-YFP-1291/Reconstruction/Temp/2016-10-03_22-58-27_488_ls.mha')
        next_surface = sitk.ReadImage('F:/NewData/2016-10-03_Thy1-YFP-1291/Reconstruction/Temp/2016-10-03_23-34-50_488_us.mha')
        _, transform2 = get_align_transform(prev_surface, next_surface,
                                            [os.path.join(ROOT_DIR,'parameters/tp_align_surface_rigid.txt'),
                                             os.path.join(ROOT_DIR,'parameters/tp_align_surface_bspline.txt')])
        out = sitk.Resample(next_surface, prev_surface, transform2)
        sitk.WriteImage(out, 'F:/chaoyu/test/m.mha')
