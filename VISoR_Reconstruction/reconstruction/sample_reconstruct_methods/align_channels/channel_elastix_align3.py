from VISoR_Brain.positioning.visor_sample import *
from VISoR_Reconstruction.misc import PARAMETER_DIR

ELASTIX_TEMP = tempfile.TemporaryDirectory()

def reconstruct1(sample_data: VISoRSample):
    elastix = sitk.ElastixImageFilter()
    elastix.SetParameterMap(sitk.ReadParameterFile(
        os.path.join(PARAMETER_DIR, 'tp_align_columns_2.txt')))
    elastix.SetOutputDirectory(ELASTIX_TEMP.name)
    elastix.SetLogToConsole(True)

    r = sample_data.raw_data
    raw_scale = sample_data.raw_data.scales['raw']
    thumb_scale = sample_data.raw_data.scales['thumbnail']
    new_transforms = {k: sitk.AffineTransform(v) for k, v in sample_data.transforms.items()}
    cum_tp = [0, 0, 0]
    tpchannel = [[0,0,0]]
    for i in range(1, len(sample_data.column_images)):

        print('Aligning column {0} to column {1}'.format(i, i - 1))
        tp = []
        threshold = 300
        x1 = sample_data.get_column_position(sample_data.column_spheres[i][0], i - 1)[1][0]
        x2 = sample_data.get_column_position(sample_data.column_spheres[i - 1][1], i)[1][0]
        while threshold > 110 and len(tp) < 1:
            ct = 100
            while ct < (len(r.columns[i]) - 100) and len(tp) < 1:
                offset = x1 - int(x1 / raw_scale) * raw_scale #- x2 + int(x2 / thumb_scale) * thumb_scale
                image1 = r.load(i - 1, ct, source_type='thumbnail')[int(x1 / thumb_scale):, :, :]
                image2 = r.load(i, ct, source_type='thumbnail')[:int(x2 / thumb_scale), :, :]
                frame1 = sitk.GetArrayFromImage(sitk.BinaryThreshold(image1[:,:,0], 0, threshold))
                frame2 = sitk.GetArrayFromImage(sitk.BinaryThreshold(image2[:,:,0], 0, threshold))
                if np.average(frame1) > 0.8 or np.average(frame2) > 0.8:
                    ct += 100
                    continue
                image1 = r.load(i - 1, (ct - 100, ct + 100), source_type='raw')[int(x1 / raw_scale):, :, :]
                image2 = r.load(i, (ct - 100, ct + 100), source_type='raw')[:int(x2 / raw_scale), :, :]
                #sitk.WriteImage(image1, 'D:/Users/chaoyu/test/1.mha')

                def pre_process(image: sitk.Image):
                    image = sitk.Clamp((sitk.Log(sitk.Cast(image, sitk.sitkFloat32)) - 4.6) * 39.4,
                                       sitk.sitkFloat32, 0, 255)
                    image.SetSpacing([raw_scale, raw_scale, 1])
                    image.SetOrigin([0, 0, 0])
                    return image

                image1 = pre_process(image1)
                image2 = pre_process(image2)
                ct += 200

                elastix.SetFixedImage(image1)
                elastix.SetMovingImage(image2)
                print('Calculating transform')

                try:
                    result = elastix.Execute()
                    #sitk.WriteImage(result, 'D:/Users/chaoyu/test/2.mha')
                except:
                    print('Failed')
                    continue
                else:
                    tp_ = elastix.GetTransformParameterMap()[0]['TransformParameters']
                    tp_ = [-float(i) for i in tp_]
                    tp.append(tp_)
            threshold = 100 + 0.5 * (threshold - 100)
        if len(tp) == 0:
            print('Failed')
            tpchannel.append([0,0,0])
            continue
        tp = np.median(tp, 0).tolist()

        p0 = [r.column_pos0[i][0] - r.pos0[0], 0, 0]
        tp = [-tp[0] - offset, -tp[1], -tp[2]]
        cum_tp = np.add(cum_tp, tp).tolist()
        tpchannel.append(cum_tp)
        #cum_tp.tolist
        #return cum_tp
        #print(tp)

        #new_transforms[i].Translate(cum_tp, False)
    return new_transforms,tpchannel


def reconstruct(sample_data: VISoRSample, reference_sample_data: VISoRSample):
    elastix = sitk.ElastixImageFilter()
    elastix.SetParameterMap(sitk.ReadParameterFile(
        os.path.join(PARAMETER_DIR, 'tp_align_channels.txt')))
    elastix.SetOutputDirectory(ELASTIX_TEMP.name)
    elastix.SetLogToConsole(True)


    reference_sample_data.load_columns()
    print("here is checkpoint1")
    prealigndata,tpchannel=reconstruct1(sample_data)
    newsampledatatransform = sample_data.transforms


    tp2 = []
    b = []
    for i in range(len(sample_data.column_images)):
        r1 = sample_data.raw_data
        r2 = reference_sample_data.raw_data

        tp = []
        threshold = 300
        while threshold > 120 and len(tp) < 1:
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
                image2 = r2.load(i, (ct - 100, ct + 100), source_type='thumbnail')

                def pre_process(image: sitk.Image):
                    image = sitk.Clamp((sitk.Log(sitk.Cast(image, sitk.sitkFloat32)) - 4.6) * 39.4,
                                       sitk.sitkFloat32, 0, 255)
                    image_ = []
                    for j in range(image.GetSize()[2]):
                        image_.append(sitk.SobelEdgeDetection(image[:,:,j]))
                    image = sitk.JoinSeries(image_)
                    return image

                image1 = pre_process(image1)
                #sitk.WriteImage(image1, 'F:/chaoyu/test/1.mha')
                image2 = pre_process(image2)
                #sitk.WriteImage(image2, 'F:/chaoyu/test/2.mha')
                ct += 200

                elastix.SetFixedImage(image1)
                elastix.SetMovingImage(image2)
                print('Aligning column {0}'.format(i, i - 1))

                try:
                    result = elastix.Execute()
                    #sitk.WriteImage(result, 'F:/chaoyu/test/3.mha')
                except:
                    print('Failed')
                    continue
                else:
                    tp_ = elastix.GetTransformParameterMap()[0]['TransformParameters']
                    tp_ = [-float(i) for i in tp_]
                    tp.append(tp_)
            threshold = 100 + 0.5 * (threshold - 100)
        if len(tp) == 0:
            print("failed")
            tp=[0,0,0]
            p0 = [r1.column_pos0[i][0] - r1.pos0[0], 0, 0]
            newsampledatatransform[i] = sitk.AffineTransform(reference_sample_data.transforms[i])
            newsampledatatransform[i].Translate(p0, True)
            newsampledatatransform[i].Translate(tp, False)
            a = newsampledatatransform[i].GetTranslation()
            b.append(np.array(a, float))
            continue
        tp = np.median(tp, 0).tolist()
        tp2.append(tp)
        p0 = [r1.column_pos0[i][0] - r1.pos0[0], 0, 0]
        newsampledatatransform[i] = sitk.AffineTransform(reference_sample_data.transforms[i])
        newsampledatatransform[i].Translate(p0, True)
        newsampledatatransform[i].Translate(tp, False)
        a = newsampledatatransform[i].GetTranslation()
        b.append(np.array(a,float))
        #print(i)
    #print("#####")
    #print(len(b))
    #tp3= tp2
    m=[]
    for i in range(len(sample_data.column_images)):
        sample_data.transforms[i] = prealigndata[i]
        sample_data.transforms[i].Translate(p0, True)
        sample_data.transforms[i].Translate(tpchannel[i], False)
        n=b[i]-np.array(sample_data.transforms[i].GetTranslation(),float)
        #print("*******")
        #print(np.array(sample_data.transforms[i].GetTranslation(),float))
        m.append(n)
    #print(m)
    m1=np.median(m,0)
    for i in range(len(sample_data.column_images)):
        m2 = sample_data.transforms[i].GetTranslation()+m1
        sample_data.transforms[i].SetTranslation(m2)
        #print(m2-b[i])



