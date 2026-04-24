from VISoR_Brain.positioning.visor_sample import *
from VISoR_Reconstruction.reconstruction.sample_reconstruct_methods.image_reconstruct.gpu_resample \
    import reconstruct as image_reconstruct
from VISoR_Reconstruction.misc import PARAMETER_DIR

ELASTIX_TEMP = tempfile.TemporaryDirectory()


def reconstruct(sample_data: VISoRSample):
    elastix = sitk.ElastixImageFilter()
    elastix.SetParameterMap(sitk.ReadParameterFile(
        os.path.join(PARAMETER_DIR, 'tp_align_stacks.txt')))
    elastix.SetOutputDirectory(ELASTIX_TEMP.name)
    elastix.SetLogToConsole(True)
    r = sample_data.raw_data

    offset = np.zeros([len(sample_data.column_images), 3], np.float64)
    overlaps = []
    for i in range(1, len(sample_data.column_images)):
        print('Aligning column {0} to column {1} (step 1)'.format(i, i - 1))
        overlap_roi = [np.subtract(r.column_pos0[i], r.pos0),
                       np.subtract(r.column_pos1[i - 1], r.pos0)]
        #overlap_roi[0][2] = -300
        prev_overlap = sitk.GetImageFromArray(
            np.clip(image_reconstruct(sample_data, overlap_roi, 4, column_index=i - 1, source='thumbnail'),
                    100, 65535) - 100)
        overlap = sitk.GetImageFromArray(
            np.clip(image_reconstruct(sample_data, overlap_roi, 4, column_index=i, source='thumbnail'),
                    100, 65535) - 100)
        #sitk.WriteImage(overlap, r'D:\workspace\test\reconstruction\20210131_ZSS_USTC_THY1-YFP_1779-180_1/' + str(i) + '.mha')
        #sitk.WriteImage(prev_overlap, r'D:\workspace\test\reconstruction\20210131_ZSS_USTC_THY1-YFP_1779-180_1/' + str(i) + '_.mha')
        overlaps.append(overlap)
        elastix.SetFixedImage(prev_overlap)
        elastix.SetMovingImage(overlap)
        print('Calculating transform')
        try:
            elastix.Execute()
            tp = elastix.GetTransformParameterMap()[0]['TransformParameters']
            tp = np.array([float(i) for i in tp]) + offset[i - 1]
            np.copyto(offset[i], tp)
        except:
            print('Failed')
            continue

    for i in range(len(sample_data.column_images)):
        sample_data.transforms[i].Translate(offset[i].tolist(), True)
    sample_data.calculate_transforms()
    sample_data.calculate_spheres()

    elastix.SetParameterMap(sitk.ReadParameterFile(
        os.path.join(PARAMETER_DIR, 'tp_align_columns_local.txt')))
    for i in range(1, len(sample_data.column_images)):
        print('Aligning column {0} to column {1} (step 2)'.format(i, i - 1))

        overlap_roi = [np.subtract(r.column_pos0[i], r.pos0),
                       np.subtract(r.column_pos1[i - 1], r.pos0)]
        x_ = None
        for threshold in [300, 200, 150, 125, 112]:
            for x in range(50, overlaps[i - 1].GetSize()[0], 50):
                if np.average(sitk.GetArrayFromImage(sitk.BinaryThreshold(overlaps[i - 1][x], 0, threshold))) < 0.8:
                    x_ = x * 4
                    break
        if x_ is None:
            print('Failed')
            continue
        overlap_roi[0][0], overlap_roi[1][0] = overlap_roi[0][0] + x_ - 100, overlap_roi[0][0] + x_ + 100
        #overlap_roi[0][2] = -300
        prev_overlap = sitk.GetImageFromArray(
            np.clip(image_reconstruct(sample_data, overlap_roi, 1, column_index=i - 1, source='raw'),
                    100, 65535) - 100)
        overlap = sitk.GetImageFromArray(
            np.clip(image_reconstruct(sample_data, overlap_roi, 1, column_index=i, source='raw'),
                    100, 65535) - 100)
        print(prev_overlap)
        sitk.WriteImage(overlap, r'D:\workspace\test\reconstruction\20210131_ZSS_USTC_THY1-YFP_1779-180_1/' + str(i) + '.mha')
        sitk.WriteImage(prev_overlap, r'D:\workspace\test\reconstruction\20210131_ZSS_USTC_THY1-YFP_1779-180_1/' + str(i) + '_.mha')
        elastix.SetFixedImage(prev_overlap)
        elastix.SetMovingImage(overlap)
        print('Calculating transform')
        try:
            elastix.Execute()
            tp = elastix.GetTransformParameterMap()[0]['TransformParameters']
            tp = np.array([float(i) for i in tp]) + offset[i - 1]
            np.copyto(offset[i], tp)
        except:
            print('Failed')
            continue

    for i in range(len(sample_data.column_images)):
        sample_data.transforms[i].Translate(offset[i].tolist(), True)
