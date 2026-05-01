from VISoR_Reconstruction.reconstruction.sample_reconstruct import reconstruct_image
from VISoR_Brain.positioning.visor_brain import VISoRSample, VISoRBrain
from VISoR_Brain.utils.ome_tiff import write_ome_tiff
import math, os, json, tifffile
import SimpleITK as sitk
import numpy as np


def region_reconstruction(brain: VISoRBrain, roi, pixel_size, output, slice_transform_file):
    with open(slice_transform_file) as fp:
        s = json.load(fp)
    slice_transform_list = {}
    for k in s['SliceTransform']:
        c = s['SliceTransform'][k]['ChannelName']
        i = s['SliceTransform'][k]['SliceID']
        if c not in slice_transform_list:
            slice_transform_list[c] = {}
        v = VISoRSample()
        v.load(os.path.join(os.path.dirname(slice_transform_file), k))
        #if v.column_source[0] != 'V':
        #    v.column_source = 'R' + v.column_source[1:]
        if not os.path.exists(v.column_source):
            print(v.column_source)
        slice_transform_list[c][i] = v

    slice_range = (brain.get_slice_position(roi[0])[0], brain.get_slice_position(roi[1])[0] + 1)
    for i in range(slice_range[0], slice_range[1]):
        transform = brain.transform(i)
        df = transform.GetDisplacementField()
        spacing = df.GetSpacing()
        slice_roi = [[max(brain.slice_spheres[i][0][j], roi[0][j]) for j in range(3)],
                     [min(brain.slice_spheres[i][1][j], roi[1][j]) for j in range(3)]]
        transform_roi = [[math.floor(roi[0][j] / spacing[j]) for j in range(2)],
                         [math.ceil(roi[1][j] / spacing[j]) for j in range(2)]]
        transform_roi[0].append(0)
        transform_roi[1].append(2)
        mesh = np.meshgrid(*[np.linspace(transform_roi[0][j] * spacing[j] + df.GetOrigin()[j],
                                         (transform_roi[1][j] - 1) * spacing[j] + df.GetOrigin()[j],
                                         transform_roi[1][j] - transform_roi[0][j]) for j in range(2, -1, -1)],
                           indexing='ij')
        df_ = sitk.GetArrayFromImage(df[transform_roi[0][0]:transform_roi[1][0],
                                     transform_roi[0][1]:transform_roi[1][1]])
        slice_image_roi = [[np.min(df_[:, :, :, j] + mesh[2 - j]) for j in range(3)],
                           [np.max(df_[:, :, :, j] + mesh[2 - j]) for j in range(3)]]
        for c in ['641']:
            if i not in slice_transform_list[c]:
                continue
            print('reconstruction {},{}'.format(i, c))
            if os.path.exists(os.path.join(output, 'BrainROIImage',
                                           'Z{:05d}_C{}.tif.txt'.format(int(math.floor(slice_roi[0][2])), c))):
                print('skip')
                continue
            slice_image_file = os.path.join(output, 'slice_roi_{}_{}.tif'.format(i, c))
            if os.path.exists(slice_image_file) and os.path.exists(slice_image_file + '.txt'):
                img = sitk.ReadImage(slice_image_file)
            else:
                img = reconstruct_image(slice_transform_list[c][i], pixel_size, slice_image_roi, source='raw')
                if not os.path.exists(os.path.dirname(slice_image_file)):
                    os.makedirs(os.path.dirname(slice_image_file))
                    with open(os.path.join(os.path.dirname(slice_image_file), 'roi.txt'), 'w') as f:
                        f.write(str(roi))
                write_ome_tiff(img, slice_image_file)
                with open(slice_image_file + '.txt', 'w'):
                    pass

            img.SetOrigin(slice_image_roi[0])
            img.SetSpacing([pixel_size, pixel_size, pixel_size])
            size = [int(
                (slice_roi[1][j] - slice_roi[0][j]) / pixel_size)
                for j in range(3)]
            res = sitk.Resample(img, size, brain.transform(i), sitk.sitkLinear,
                                slice_roi[0],
                                [pixel_size, pixel_size, pixel_size])
            res = sitk.DivideFloor(res, 16) * 16
            path = [os.path.join(output, 'BrainROIImage',
                                 'Z{:05d}_C{}.tif'.format(int(math.floor(slice_roi[0][2] / pixel_size) + j), c))
                    for j in range(size[2])]
            if not os.path.exists(os.path.dirname(path[0])):
                os.makedirs(os.path.dirname(path[0]))
            for j in range(len(path)):
                im = sitk.GetArrayFromImage(res[:, :, j])
                tifffile.imwrite(path[j], im, compress=1)
            with open(path[0] + '.txt', 'w'):
                pass


if __name__ == '__main__':
    brain = VISoRBrain('E:/brains/RM005/Reconstruction/BrainTransform/visor_brain.txt')
    region_reconstruction(brain, [[33200, 38300, 50700], [38200, 46600, 56700]], 1,
                          'E:/brains/RM005/Analysis/ROIReconstruction',
                          'E:/brains/RM005/Reconstruction/SliceTransform/SliceTransform.json')
