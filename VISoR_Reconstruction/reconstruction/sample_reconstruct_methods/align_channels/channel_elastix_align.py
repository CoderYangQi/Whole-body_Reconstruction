from VISoR_Brain.positioning.visor_sample import *
from VISoR_Reconstruction.misc import PARAMETER_DIR

ELASTIX_TEMP = tempfile.TemporaryDirectory()

def reconstruct(sample_data: VISoRSample, reference_sample_data: VISoRSample):
    elastix = sitk.ElastixImageFilter()
    elastix.SetParameterMap(sitk.ReadParameterFile(
        os.path.join(PARAMETER_DIR, 'tp_align_channels.txt')))
    elastix.SetOutputDirectory(ELASTIX_TEMP.name)
    elastix.SetLogToConsole(True)

    reference_sample_data.load_columns()

    for i in range(len(sample_data.column_images)):
        r1 = sample_data.raw_data
        r2 = reference_sample_data.raw_data

        tp = []
        threshold = 300
        while threshold > 110 and len(tp) < 1:
            ct = 100
            while ct < len(r1.columns[i]) - 100 and len(tp) < 1:
                image1 = r1.load(i, ct, source_type='thumbnail')
                image2 = r2.load(i, ct, source_type='thumbnail')
                frame1 = sitk.GetArrayFromImage(sitk.BinaryThreshold(image1[:,:,0], 0, threshold))
                frame2 = sitk.GetArrayFromImage(sitk.BinaryThreshold(image2[:,:,0], 0, threshold))
                if np.average(frame1) > 0.95 or np.average(frame2) > 0.95:
                    ct += 100
                    continue
                image1 = r1.load(i, (ct - 100, ct + 100), source_type='thumbnail')
                #sitk.WriteImage(image1, 'F:/chaoyu/test/1.mha')
                image2 = r2.load(i, (ct - 100, ct + 100), source_type='thumbnail')
                #sitk.WriteImage(image2, 'F:/chaoyu/test/2.mha')

                def pre_process(image: sitk.Image):
                    image = sitk.Clamp((sitk.Log(sitk.Cast(image, sitk.sitkFloat32)) - 4.6) * 39.4,
                                       sitk.sitkFloat32, 0, 255)
                    image_ = []
                    for j in range(image.GetSize()[2]):
                        image_.append(sitk.SobelEdgeDetection(image[:,:,j]))
                    image = sitk.JoinSeries(image_)
                    return image

                image1 = pre_process(image1)
                #sitk.WriteImage(image1, 'F:/chaoyu/test/1_.mha')
                #image2_ = sitk.Image(image2)
                image2 = pre_process(image2)
                #sitk.WriteImage(image2, 'F:/chaoyu/test/2_.mha')
                ct += 200

                elastix.SetFixedImage(image1)
                elastix.SetMovingImage(image2)
                print('Aligning column {0}'.format(i, i - 1))

                try:
                    result = elastix.Execute()
                    #sitk.WriteImage(result, 'F:/chaoyu/test/3.mha')
                    #result = sitk.Transformix(image2_, elastix.GetTransformParameterMap())
                    #sitk.WriteImage(result, 'F:/chaoyu/test/3_.mha')
                except:
                    print('Failed')
                    continue
                else:
                    tp_ = elastix.GetTransformParameterMap()[0]['TransformParameters']
                    tp_ = [-float(i) for i in tp_]
                    tp.append(tp_)
            threshold = 100 + 0.5 * (threshold - 100)
        if len(tp) == 0:
            continue
        tp = np.median(tp, 0).tolist()
        #p0 = [r1.column_pos0[i][0] - r1.pos0[0], 0, 0]
        #tp[2] = tp[2] * 0.5
        sample_data.transforms[i] = sitk.AffineTransform(reference_sample_data.transforms[i])
        #sample_data.transforms[i].Translate(p0, True)
        sample_data.transforms[i].Translate(tp, False)
