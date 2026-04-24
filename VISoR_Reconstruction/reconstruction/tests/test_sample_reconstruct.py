from VISoR_Reconstruction.reconstruction.sample_reconstruct import *
import unittest
import random, csv
import torch
import torch.nn.functional as F
from VISoR_Brain.utils.ome_tiff import write_ome_tiff


class SampleReconstructTestCase(unittest.TestCase):

    def test_get_all_methods(self):
        print(get_all_methods())

    def test_reconstruct_sample(self):
        raw = RawData(os.path.join(TEST_DATA_DIR, 'visor12_data/1_010/VGLUT_SP6_2_1_010_2.flsm'))
        sl = reconstruct_sample(raw, {'stitch': 'stage_position'})
        sl.save(os.path.join(TEST_DATA_DIR, 'visor12_data/VGLUT_SP6_2/488/VGLUT_SP6_2_1_010_2.tar'))

    def test_elastix_align(self):
        raw = RawData(os.path.join(TEST_DATA_DIR, 'visor12_data/1_010/VGLUT_SP6_2_1_010_2.flsm'))
        sl = reconstruct_sample(raw, {'stitch': 'elastix_align'})
        sl.save(os.path.join(TEST_DATA_DIR, 'visor12_data/VGLUT_SP6_2/488/VGLUT_SP6_2_1_010_2.tar'))

    def test_elastix_align2(self):
        raw = RawData(r'Z:\SIAT_SIAT\BiGuoqiang\Mouse_Brain\20210131_ZSS_USTC_THY1-GFP_11_1\Data\488nm_10X\1_024/USTC_THY1-GFP_11_1_024_1.flsm')
        #raw = RawData(os.path.join(TEST_DATA_DIR, 'visor12_data/1_010/VGLUT_SP6_2_1_010_2.flsm'))
        sl = reconstruct_sample(raw, {'stitch': 'elastix_align2'}, flip=False)
        #sl.save(os.path.join(TEST_DATA_DIR, 'results/VGLUT_SP6_2_1_010_2.txt'))
        sl.save(os.path.join(TEST_DATA_DIR, 'results/t.txt'))

    def test_reconstruct_image(self):
        #sl = load_visor_sample(os.path.join(TEST_DATA_DIR, 'results/VGLUT_SP6_2_1_010_2.txt'))
        sl = load_visor_sample(os.path.join(TEST_DATA_DIR, 'results/t.txt'))
        roi = sl.sphere
        image = reconstruct_image(sl, 4, roi, 'gpu_resample',source='thumbnail', blend_method='right_side')
        write_ome_tiff(image, os.path.join(TEST_DATA_DIR, 'results/test_reconstruct_image.tif'))
        #sitk.WriteImage(image, os.path.join(TEST_DATA_DIR, 'visor12_data/VGLUT_SP6_2/488/VGLUT_SP6_2_1_010_2.tif'))

    '''
    def test_reconstruct_image_gpu(self):
        sl = load_visor_sample(os.path.join(TEST_DATA_DIR, 'visor12_data/VGLUT_SP6_2/488/VGLUT_SP6_2_1_010_2.tar'))
        roi = [[sl.sphere[0][0], sl.sphere[0][1], sl.sphere[0][2]],
               [(sl.sphere[1][0] + sl.sphere[0][0]) / 2, sl.sphere[1][1], sl.sphere[1][2]]]
        #roi = sl.sphere
        image = reconstruct_image(sl, 1, roi, 'gpu_resample',source='raw')
        sitk.WriteImage(image, os.path.join(TEST_DATA_DIR, 'visor12_data/VGLUT_SP6_2/488/VGLUT_SP6_2_1_010_2.tif'))
    '''


    def test_eval_reconstruct_sample(self):
        shift = 10
        block_size = 50
        from VISoR_Brain.format.visor_data import VISoRData
        dataset = VISoRData(r'Y:\SIAT_SIAT\LuZhonghua\CrabeatingMacaque_Brain_CM001/reconstruction_local.visor')
        output_path = r'D:\workspace\test\eval_reconstruct_sample'
        slice_transform_list = dataset.slice_transform
        min_point = []
        min_ncc = []
        block_names = []
        for idx in slice_transform_list['1']:
            if idx not in [80, 84, 88, 92, 96, 100]:
                continue
            sl = VISoRSample(slice_transform_list['1'][idx])
            n = 0
            '''
            while n < (len(sl.transforms) - 1) ** 2 / 4:
                n_col = random.randint(1, len(sl.transforms) - 1)
                p0 = [random.uniform(sl.sphere[0][0] + shift, sl.sphere[1][0] - block_size - shift),
                      random.uniform(sl.column_spheres[n_col][0][1] + shift,
                                     sl.column_spheres[n_col - 1][1][1] - block_size - shift),
                      random.uniform(sl.sphere[0][2] + shift, sl.sphere[1][2] - block_size - shift)]
            '''
            for n_col, x_ in [(n, x__)
                              for n in [5, 8, 11, 14]
                              for x__ in range(int(sl.sphere[0][0] + 2000), int(sl.sphere[1][0] - 2000), 500)]:
                p0 = [x_,
                      (sl.column_spheres[n_col][0][1] + sl.column_spheres[n_col - 1][1][1] - block_size) / 2,
                      (sl.column_spheres[n_col][0][2] + sl.column_spheres[n_col - 1][1][2] - block_size) / 2]
                print(p0)
                roi = [[np.round(i) for i in p0], [np.round(i) + block_size for i in p0]]
                img_l = reconstruct_image(sl, 1, roi, 'gpu_resample', source='raw', blend_method='left_side',
                                          rawdata=RawData(dataset.acquisition_results['1'][idx]))
                img_l_ = np.float32(img_l)
                img_l = sitk.GetImageFromArray(img_l)
                if np.average(img_l_) < 130:
                    continue
                block_name = '{}_{}_{}_{}'.format(idx, n_col, int(p0[0]), int(p0[2]))
                block_names.append(block_name)
                sitk.WriteImage(img_l, os.path.join(output_path, '{}_l.mha'.format(block_name)))
                img_l = torch.Tensor(img_l_)
                roi = [[i - shift for i in roi[0]], [i + shift for i in roi[1]]]
                img_r = reconstruct_image(sl, 1, roi, 'gpu_resample', source='raw', blend_method='right_side')
                img_r = sitk.GetImageFromArray(img_r)
                sitk.WriteImage(img_r, os.path.join(output_path, '{}__r.mha'.format(block_name)))
                img_r = torch.Tensor(np.float32(sitk.GetArrayFromImage(img_r)))[None,]
                ncc = []
                norm_l = (img_l - torch.mean(img_l)).contiguous().view((1, block_size ** 3, 1))
                s_norm_l = torch.mean(norm_l ** 2)
                for z in range(0, 2 * shift + 1):
                    m = F.unfold(img_r[:, z: z + block_size, :, :], (block_size, block_size))
                    #l = img_l.contiguous().view((1, block_size ** 3, 1))
                    #m = torch.mean((m - l) ** 2, 1)
                    norm_r = m - torch.mean(m, 1)
                    m = torch.mean(norm_r * norm_l, 1) / torch.sqrt(s_norm_l * (torch.mean(norm_r ** 2, 1)))
                    ncc.append(m.view((2 * shift + 1, 2 * shift + 1)))
                ncc = -torch.stack(ncc).cpu().numpy()
                minp = np.unravel_index(np.argmin(ncc), ncc.shape)
                minn = ncc[minp[0], minp[1], minp[2]]
                minps = list(minp)
                for i in range(3):
                    if 1 < minp[i] < 20:
                        p1, p_1 = list(minp), list(minp)
                        p1[i] += 1
                        p_1[i] -= 1
                        n1, n_1 = ncc[p1[0], p1[1], p1[2]], ncc[p_1[0], p_1[1], p_1[2]]
                        minps[i] = minp[i] + (n1 - n_1) / (4 * minn - 2 * (n1 + n_1))
                min_ncc.append(minn)
                min_point.append(minps)
                print(minp, min_ncc)
                sitk.WriteImage(sitk.GetImageFromArray(ncc), os.path.join(output_path, '{}_ncc.tif'.format(block_name)))
                n += 1
        with open(os.path.join(output_path, 'summary.csv'), 'w') as file:
            line = 'name,x,y,z,distance,ncc\n'
            file.write(line)
            for j in range(len(min_point)):
                p = [i - shift for i in min_point[j]]
                line = '{},{},{},{},{},{}\n'.format(block_names[j], p[2], p[1], p[0], np.sqrt(np.sum(np.power(p, 2))), min_ncc[j])
                file.write(line)


    def test_overlap_cv2_align(self):
        raw = RawData(os.path.join(TEST_DATA_DIR, 'visor12_data/1_010/VGLUT_SP6_2_1_010_2.flsm'))
        sl = reconstruct_sample(raw, {'stitch': 'overlap_cv2_align'})
        sl.save(os.path.join(TEST_DATA_DIR, 'visor12_data/VGLUT_SP6_2/488/VGLUT_SP6_2_1_010_2.tar'))
        image = reconstruct_image(sl, 4, sl.sphere, source='thumbnail')
        sitk.WriteImage(image, os.path.join(TEST_DATA_DIR, 'visor12_data/VGLUT_SP6_2/488/VGLUT_SP6_2_1_010_2.tif'))
'''
if __name__ == '__main__':
    t = SampleReconstructTestCase()
    import cProfile
    cProfile.run('t.test_reconstruct_image()')
    #t.test_elastix_align2()
    #t.test_eval_reconstruct_sample()
#'''
