from VISoR_Reconstruction.reconstruction.brain_reconstruct import *
import unittest, json


class BrainReconstructTestCase(unittest.TestCase):

    def test_surface_segmentation(self):
        image = sitk.ReadImage(os.path.join(TEST_DATA_DIR, 'visor12_data/VGLUT_SP6_2/488/VGLUT_SP6_2_1_010_2.tif'))
        umap, lmap = calc_surface_height_map(image)
        sitk.WriteImage(umap, 'F:/chaoyu/test/flatten/uz.mha')
        sitk.WriteImage(lmap, 'F:/chaoyu/test/flatten/lz.mha')
        us, ls = extract_surface(image, umap, lmap)
        sitk.WriteImage(us, 'F:/chaoyu/test/flatten/us.mha')
        sitk.WriteImage(ls, 'F:/chaoyu/test/flatten/ls.mha')

    def test_process_transform(self):
        with open('F:/NewData/2016-10-03_Thy1-YFP-1291/Reconstruction/ReconstructionInput.json') as f:
            doc = json.load(f)['tasks']['process_transforms_TY_1291_1291']
        input_image = {i: sitk.ReadImage(doc['input_targets'][i]['path']) for i in doc['input_targets']}
        param = doc['parameters']
        tf = process_transforms_(input_image, **param)
        output_image = [i['path'] for i in doc['output_targets']]
        for i in range(len(output_image)):
            sitk.WriteImage(tf[i], output_image[i])

    def test_create_brain(self):
        with open('D:/Hao/Data/converted/cfos-C2_2652/Reconstruction/ReconstructionInput.json') as f:
            doc = json.load(f)['tasks']['create_brain_cfos-C2_2652']
        input_ = {}
        for k, v in doc['input_targets'].items():
            if v['type'] == 'image':
                input_[k] = sitk.ReadImage(v['path'])
            else:
                input_[k] = VISoRSample()
                input_[k].load(v['path'])
        param = doc['parameters']
        output = doc['output_targets'][0]['path']
        br = create_brain_(input_, **param, output_path=output)
        br.save(output)


    def test_align_surfaces1(self):
        prev_surface = sitk.ReadImage(r'C:\Users\chaoyu\Documents\test\spinal_cord/MCZ_368_1_099_488nm_10X_ls.mha')
        next_surface = sitk.ReadImage(r'C:\Users\chaoyu\Documents\test\spinal_cord/MCZ_368_1_100_488nm_10X_us.mha')
        #ref = sitk.ReadImage('C:/Users/chaoyu/Documents/projects/VISoR-data-analysis/VISoR_Brain/data/macaque_brain_template/099.tif')
        d1, d2 = align_surfaces(prev_surface=prev_surface, next_surface=next_surface, method='elastix', ref_img=None,
                                outside_brightness=10, ref_scale=4, ref_size=[3500, 2500])
        sitk.WriteImage(sitk.DisplacementFieldJacobianDeterminant(d1), r'C:\Users\chaoyu\Documents\test\spinal_cord/d1.mha')
        sitk.WriteImage(sitk.DisplacementFieldJacobianDeterminant(d2), r'C:\Users\chaoyu\Documents\test\spinal_cord/d2.mha')
        t1 = sitk.DisplacementFieldTransform(sitk.Image(d1))
        t2 = sitk.DisplacementFieldTransform(sitk.Image(d2))
        out1 = sitk.Resample(prev_surface, d1, t1)
        out2 = sitk.Resample(next_surface, d2, t2)
        sitk.WriteImage(out1, r'C:\Users\chaoyu\Documents\test\spinal_cord/1.mha')
        sitk.WriteImage(out2, r'C:\Users\chaoyu\Documents\test\spinal_cord/2.mha')


    def test_align_surfaces2(self):
        prev_surface = [sitk.ReadImage('F:/TEST_DATA/Mouse_Brain/20180914_ZMN_WH_438_1_1/Reconstruction/Temp/WH_438_1_015_488nm_10X_ls.mha'),
                        sitk.ReadImage('F:/TEST_DATA/Mouse_Brain/20180914_ZMN_WH_438_1_1/Reconstruction/Temp/WH_438_1_015_552nm_10X_ls.mha')]
        next_surface = [sitk.ReadImage('F:/TEST_DATA/Mouse_Brain/20180914_ZMN_WH_438_1_1/Reconstruction/Temp/WH_438_1_016_488nm_10X_us.mha'),
                        sitk.ReadImage('F:/TEST_DATA/Mouse_Brain/20180914_ZMN_WH_438_1_1/Reconstruction/Temp/WH_438_1_016_552nm_10X_us.mha')]
        #ref = sitk.ReadImage('C:/Users/chaoyu/Documents/projects/VISoR-data-analysis/VISoR_Brain/data/macaque_brain_template/099.tif')
        d1, d2 = align_surfaces('elastix_multichannel',
                                prev_surface_0=prev_surface[0], prev_surface_1=prev_surface[1],
                                next_surface_0=next_surface[0], next_surface_1=next_surface[1],
                                ref_img=None, outside_brightness=10, ref_scale=62.5, ref_size=[3000, 2000])
        sitk.WriteImage(sitk.DisplacementFieldJacobianDeterminant(d1), 'F:/chaoyu/test/d1.mha')
        sitk.WriteImage(sitk.DisplacementFieldJacobianDeterminant(d2), 'F:/chaoyu/test/d2.mha')
        t1 = sitk.DisplacementFieldTransform(sitk.Image(d1))
        t2 = sitk.DisplacementFieldTransform(sitk.Image(d2))
        out1 = sitk.Resample(prev_surface[0], d1, t1)
        out2 = sitk.Resample(next_surface[0], d2, t2)
        sitk.WriteImage(out1, 'F:/chaoyu/test/1.mha')
        sitk.WriteImage(out2, 'F:/chaoyu/test/2.mha')
