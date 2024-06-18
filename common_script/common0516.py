'''
@ yangqi
修改了 separate 里面的 forbid location 增加了对 next image 的空缺判断；

'''

from common_script.yq_elastix_files import *
from VISoR_Reconstruction.misc import PARAMETER_DIR
import numpy as np
import cv2
import json
from common_script.VISoR_Data import VISoRData
import os , re,time
import argparse

def get_files(image_root, extend_name):
    image_files_path = os.listdir(image_root)
    path_temp = []
    for i in image_files_path:
        fname, ext = os.path.splitext(i)
        if ext == extend_name:
            i = os.path.join(image_root, i)
            path_temp.append(i)
        pass
    image_files_path = path_temp
    return  image_files_path

def create_folder(name):
    if not os.path.exists(name):
        os.mkdir(name)

def pointsBoundBox(points):
    array = np.array(points)
    lefttop = array.min(axis=0)
    rightbottom = array.max(axis=0)
    return lefttop,rightbottom

class SliceData:
    def __init__(self,index,transform_path):
        self.m_sliceIndex = index
        self.path = transform_path
        self.info = None
        self.sphere = None
        self.sliceOrigin = None
        self.sliceOrigin2 = None
        self.ReadInfo()
    def ReadInfo(self):
        with open(self.path) as f:
            self.info = json.load(f)
        self.sphere = self.info['sphere']
        self.sliceOrigin = self.sphere[0]
        self.sliceOrigin2 = self.sphere[1]


class Reconstruction_Point():
    def __init__(self,thickness = 500):
        self.slice_images = None
        self.reconstruction = None
        self.points = None
        self.sizeF = None
        self.lefttop = None
        self.rightbottom = None
        self.tops = None
        self.thickness = 400
        self.channel = '488nm'
        self.basePath = None
        self.uZ = None
        self.lZ = None
        self.internal_pixel_size = None
        self.name = None
        self.channels = None
        self.slices = None
        self.pxSize = None
    def init(self):
        parser = argparse.ArgumentParser(description="Demo of add tfm")
        parser.add_argument('-p', '--path',
                            # default=r'Z:\Data\E\E-123\Reconstruction')
                            default = r'D:\USERS\yq\TH2_Reconstruction\Reconstruction')
        parser.add_argument('-c', '--channel',
                            default='561nm')
        parser.add_argument("-r",'--reconstruction',
                            # default=r"Z:\Data\E\E-123\1108 Reconstruction.visor")
                            default = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction.visor")
        parser.add_argument("-padding", '--paddingZ',
                            default=40)
        args = parser.parse_args(args=[])
        basePath = args.path
        self.basePath = basePath
        self.channel = args.channel
        self.reconstruction = VISoRData(args.reconstruction)
        self.paddingZ = args.paddingZ
        slicePath=basePath+"/SliceTransform/"
        outputPath=basePath+"/BrainTransform/transforms/"
        visorPath=args.reconstruction
        parametersPath = basePath+"/Parameters.json"
        with open(visorPath) as f:
            info = json.load(f)
        #  get thickness
        a = info["Acquisition Results"]
        # flsm0 = a[0]
        # flsm1 = a[1]
        # flsmlist = 'FlsmList'
        # flsm0_path = os.path.join(os.path.abspath(os.path.join(basePath, "..")),flsm0[flsmlist][0])
        # flsm1_path = os.path.join(os.path.abspath(os.path.join(basePath, "..")),flsm1[flsmlist][0])
        # raw0 = RawData(flsm0_path)
        # raw1 = RawData(flsm1_path)
        # lefttop_z = 'lefttop_z'
        # thickness = int((raw0.info[lefttop_z] - raw1.info[lefttop_z])*1000)
        thickness = 400
        self.thickness = thickness
        # thickness = self.thickness
        print("thickness is ",thickness)
        # print(flsm0_path)
        with open(parametersPath) as f:
            info = json.load(f)
        pxSize=info["internal_pixel_size"]
        self.pxSize = pxSize
        self.internal_pixel_size = info["internal_pixel_size"]
        print('pxSize is ',pxSize)
        info["slice_thickness"]=thickness
        #  slice list
        slice_list = get_files(slicePath,'.txt')
        slice_index = 1 # TODO 需要添加名字读取出index
        slices = {}
        points = []

        channel = self.channel
        temp_list = []
        #  目前只考虑488nm的图像
        for i in slice_list:
            if re.search(channel,i):
                print(i)
                temp_list.append(i)
        slice_list = temp_list
        self.channels = self.reconstruction.channels  # get channels name

        # slice_list = self.reconstruction.slice_transform['2']

        #
        # for i in (slice_list):
        #     path = slice_list[i]
        #     print(path)
        #     slice = SliceData(i, path)
        #     slices[i] = slice
        #     points.append(slice.sliceOrigin)
        #     points.append(slice.sliceOrigin2)
        #     # print("path is : ",i)
        #     slice_index += 1

        for i in range(len(slice_list)):
            path = slice_list[i]
            print(path)
            slice = SliceData(i, path)
            slices[i] = slice
            points.append(slice.sliceOrigin)
            points.append(slice.sliceOrigin2)
            # print("path is : ",i)
            slice_index += 1

        lefttop,rightbottom = pointsBoundBox(points)
        sizeF = (rightbottom - lefttop)/pxSize
        width = int(sizeF[0])
        height = int(sizeF[1])

        tops = []
        bottoms = []
        pDispTops = []
        uZ = []
        lZ = []
        # create 形变场
        index = 0
        for i in slices:
            # 仅仅尝试 index = 6
            # if i != 6:
            #     continue
            slice = slices[i]
            paddingZ = self.paddingZ
            offsetZ = (slice.m_sliceIndex - 1) * thickness
            bottom = slice.sliceOrigin2[2] - paddingZ
            top = bottom - thickness
            # pDispTop = top - offsetZ
            pDispTop = top
            tops.append(top)
            bottoms.append(bottom)
            pDispTops.append(pDispTop)
            U_height = (top - points[index*2][2])/pxSize
            uZ.append(U_height)
            lZ.append(U_height + thickness/pxSize)
            index += 1

        # SliceImagePath = os.path.join(basePath,'SliceImage','4.0','488nm')
        SliceImagePath = os.path.join(basePath, 'SliceImage', '4.0')
        slice_images = get_files(SliceImagePath,'.tif')
        temp_list = []
        #  目前只考虑488nm的图像
        for i in slice_images:
            if re.search(channel, i):
                print(i)
                temp_list.append(i)
        slice_images = temp_list

        self.slice_images = slice_images
        self.points = points
        self.sizeF = sizeF
        self.lefttop = lefttop
        self.rightbottom = rightbottom
        self.tops = tops
        self.bottoms = bottoms
        self.uZ = uZ
        self.lZ = lZ
        # self.thickness = thickness
        self.slice_images = slice_images
        self.slices = slices
        self.name = self.reconstruction.name
    def create_750(self):
        pxSize = self.pxSize
        slices = self.slices
        thickness = self.thickness
        points = self.points
        tops = []
        bottoms = []
        pDispTops = []
        uZ = []
        lZ = []
        index = 0
        for i in slices:
            # 仅仅尝试 index = 6
            # if i != 6:
            #     continue
            slice = slices[i]
            paddingZ = self.paddingZ
            range_z = slice.sliceOrigin2[2] - slice.sliceOrigin[2]
            if range_z-paddingZ > 750:
                bottom = slice.sliceOrigin[2] + 750
            else:
                bottom = slice.sliceOrigin2[2] - paddingZ

            offsetZ = (slice.m_sliceIndex - 1) * thickness
            top = bottom - thickness
            # pDispTop = top - offsetZ
            pDispTop = top
            tops.append(top)
            bottoms.append(bottom)
            pDispTops.append(pDispTop)
            U_height = (top - points[index*2][2])/pxSize
            uZ.append(U_height)
            lZ.append(U_height + thickness/pxSize)
            index += 1
        self.uZ = uZ
        self.lZ = lZ
        pass
def Preprocess(surface,threshold):
    # if img_path == None:
    #     return None
    # surface = sitk.ReadImage(img_path)
    # threshold = 120
    surface = sitk.Threshold(surface, threshold, 65535, threshold)
    back_log_value = np.log(threshold)
    # back_log_value = 0
    surface = sitk.Clamp((sitk.Log(sitk.Cast(surface + 1, sitk.sitkFloat32)) - back_log_value) * 39.4,
                         sitk.sitkUInt8, 0, 255)
    # surface = sitk.Clamp((sitk.Log(sitk.Cast(surface + 1, sitk.sitkFloat32)) - back_log_value) * 39.4,
    #                      sitk.sitkUInt8, 0, 255)
    return surface


def GetBottom(size,spacing = 10):
    z = size[2]
    if z - 4 > 75:
        z = 75
    else:
        z = z - 4

    return z

def GetBottom_4um(size,spacing = 10):
    z = size[2]
    if z - 5 > 175:
        z = 175
    else:
        z = z - 5

    return z

def RefinePos(img,sizeF,point,Lefttop,spacing):
    # init transform
    dimension = 3
    output_pixel_size = 10
    sizeImg = img.GetSize()
    sizeF = [sizeF[0], sizeF[1], sizeImg[2]]
    translation = sitk.TranslationTransform(dimension)
    translation.SetParameters((0, 0, 0))
    img.SetOrigin(point)
    img.SetSpacing(spacing)

    img = sitk.Resample(img,sizeF,translation,sitk.sitkLinear, Lefttop,spacing)


    return img

def fill_outside_yq(img, value: int):
    # img = sitk.GetArrayFromImage(img)
    img[0, 0] = 0
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    cv2.floodFill(img,
                  mask,
                  (0, 0), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    img[img.shape[0] - 1, 0] = 0
    cv2.floodFill(img,
                  mask,
                  (0, img.shape[0] - 1), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    img[img.shape[0] - 1, img.shape[1] - 1] = 0
    cv2.floodFill(img,
                  mask,
                  (img.shape[1] - 1, img.shape[0] - 1), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    img[0, img.shape[1] - 1] = 0
    cv2.floodFill(img,
                  mask,
                  (img.shape[1] - 1, 0), value, value, value, cv2.FLOODFILL_FIXED_RANGE)
    # img = sitk.GetImageFromArray(img)
    return img
# todo 在 SeparateBlock 中调用 用来计算小块block的xyz位移
def CalBlock(prev_surface, next_surface,spacing, ref_img: sitk.Image = None, prev_points=None, next_points=None,
                   outside_brightness=2, nonrigid=True, ref_size=None, ref_scale=1, use_rigidity_mask=False, **kwargs):
    size = prev_surface.GetSize()
    temp = sitk.GetArrayFromImage(prev_surface)
    if len(size) == 2:
        temp[:, :] = fill_outside_yq(temp[:, :], outside_brightness)
    else:
        for i in range(size[2]):
            temp[i, :, :] = fill_outside_yq(temp[i, :, :], outside_brightness)
    prev_surface = sitk.GetImageFromArray(temp)

    size = next_surface.GetSize()
    temp = sitk.GetArrayFromImage(next_surface)
    if len(size) == 2:
        temp[:, :] = fill_outside_yq(temp[:, :], outside_brightness)
    else:
        for i in range(size[2]):
            temp[i, :, :] = fill_outside_yq(temp[i, :, :], outside_brightness)
    next_surface = sitk.GetImageFromArray(temp)

    # prev_surface = fill_outside_yq(prev_surface, outside_brightness)
    # next_surface = fill_outside_yq(next_surface, outside_brightness)
    '''
    只是做 affine 效果查看
    '''
    # justify the spcaing of imgs
    prev_surface.SetSpacing(spacing)
    next_surface.SetSpacing(spacing)

    next_result, transform2 = get_align_transform(prev_surface, next_surface,
                                                  [os.path.join(PARAMETER_DIR, 'p_3D_translation.txt')])
    # next_result, transform2 = get_align_transform(prev_surface, next_surface,
    # [os.path.join(PARAMETER_DIR, 'p_affine.txt')],
    # rigidity_mask=rigidity_mask)
    # next_df = sitk.TransformToDisplacementField(transform2,
    #                                             sitk.sitkVectorFloat64,
    #                                             ref_img.GetSize(),
    #                                             ref_img.GetOrigin(),
    #                                             ref_img.GetSpacing(),
    #                                             ref_img.GetDirection())
    return prev_surface, next_result, transform2
# todo 计算区域划分，forbid 一些评分低的点
def SeparateBlock(img,up_img,down_img,spacing,bottom1,end2,roi, slices_index,
                  block_size = 250,sub_block = 250, tempName = 'th2_111_112'):
    if img == None:
        img = sitk.MaximumProjection(down_img,projectionDimension=2)[:,:,0]
    up_img.SetSpacing(spacing)
    up_img.SetOrigin([0,0,0])
    down_img.SetSpacing(spacing)
    down_img.SetOrigin([0, 0, 0])
    size = img.GetSize()
    row = int(np.floor(size[0]/block_size))
    col = int(np.floor(size[1]/block_size))
    vector_points = np.zeros((row,col,3))
    forbid_points = np.zeros((row,col))
    # todo 计算平均的评分指标结果 然后用阈值进行分割 ban掉无用区间
    back_brightness = 120
    for i in range(row):
        for j in range(col):
            temp = sitk.GetArrayFromImage(img[i*block_size : (i+1)*block_size,
                                          j*block_size:(j+1)*block_size])
            sub_temp = temp[block_size - sub_block:, block_size - sub_block:]
            bool_temp = sub_temp > back_brightness
            int_temp = np.array(bool_temp,dtype=np.int)

            holow_scale = np.mean(np.mean(int_temp))
            forbid_points[i, j] = holow_scale

            # print("")
    # print("")

    # todo 计算数据
    size1 = up_img.GetSize()
    size2 = down_img.GetSize()
    # 计算高度 圈定大概的数据范围
    # bottom1 = GetBottom(size1)
    # bottom2 = GetBottom(size2)
    # end2 = bottom2 - 40
    up_img = Preprocess(up_img,120)
    down_img = Preprocess(down_img,120)
    # MaxImage1 = up_img[:, :, bottom1 - 30:bottom1]
    # origin = [0, 0, 0]
    # MaxImage1.SetOrigin(origin)
    # MaxImage2 = down_img[:, :, end2 - 30:end2]
    # MaxImage2.SetOrigin(origin)

    # spacing = [10,10,10]
    tf_pars = []
    pos = []
    # i,j = 5,2
    for i in range(row):
        for j in range(col):

            # if i != 5 or col != 2:
            #     continue

            # print("i is {}; j is {}".format(i,j))
            if forbid_points[i, j]>0.4:
                start = time.time()
                up_temp = up_img[i*block_size : (i+1)*block_size,j*block_size:(j+1)*block_size,
                          roi[0][0] :roi[0][1]]
                down_temp = down_img[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size,
                          roi[1][0] :roi[1][1]]
                # down_temp = down_img[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size,
                #             end2 - interval:end2]
                sub_up = up_temp[block_size-sub_block:,block_size-sub_block:,:]
                sub_down = down_temp[block_size - sub_block:, block_size - sub_block:, :]

                # todo 判断是否有过多的数据缺失
                max_sub_down = sitk.MaximumProjection(sub_down[:,:,:(roi[0][1] - roi[0][0])//2], projectionDimension=2)[:, :, 0]
                hollow_scale = np.mean(np.mean(max_sub_down))
                if hollow_scale < 0.4:
                    continue

                # todo 找出非背景图像的表面 start 经过测试 还是算了，现在尝试拼接另一个表面数据做测试
                # drop, drop_hollow_scale = FindUsefulSurface(sub_down)
                # down_temp = down_img[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size,
                #             end2 - interval + drop:end2 + drop]
                # sub_down = down_temp[block_size - sub_block:, block_size - sub_block:, :]


                # todo 找出非背景图像的表面 end

                origin = [0, 0, 0]
                # up_temp.SetOrigin(origin)
                # up_temp.SetSpacing(spacing)
                # down_temp.SetOrigin(origin)
                # down_temp.SetSpacing(spacing)
                # todo 使用 200 * 200 的 右下角的块进行测试
                sub_up.SetOrigin(origin)
                sub_up.SetSpacing(spacing)
                sub_down.SetOrigin(origin)
                sub_down.SetSpacing(spacing)
                # create file folder
                create_folder(os.path.join(r"D:\USERS\yq\code\cal_overlap\Refine",tempName))

                max1 = sitk.MaximumProjection(sub_up,projectionDimension=2)[:,:,0]
                max2 = sitk.MaximumProjection(sub_down, projectionDimension=2)[:, :, 0]
                # sitk.WriteImage(max1,"Refine/"+ tempName +str(i)+"_"+str(j)+"up_temp.tif")
                # sitk.WriteImage(max2,"Refine/"+ tempName +str(i)+"_"+str(j)+"down_temp.tif")
                max1 = sitk.MaximumProjection(sub_up, projectionDimension=0)[0, :, :]
                max2 = sitk.MaximumProjection(sub_down, projectionDimension=0)[0, :, :]
                # sitk.WriteImage(max1, "Refine/" + tempName + str(i) + "_" + str(j) + "yoz_up_temp.tif")
                # sitk.WriteImage(max2, "Refine/" + tempName + str(i) + "_" + str(j) + "yoz_down_temp.tif")
                max1 = sitk.MaximumProjection(sub_up, projectionDimension=1)[:, 0, :]
                max2 = sitk.MaximumProjection(sub_down, projectionDimension=1)[:, 0, :]
                # sitk.WriteImage(max1, "Refine/" + tempName + str(i) + "_" + str(j) + "xoz_up_temp.tif")
                # sitk.WriteImage(max2, "Refine/" + tempName + str(i) + "_" + str(j) + "xoz_down_temp.tif")
                block_save_path = os.path.join(r"D:\USERS\yq\code\cal_overlap\Refine", tempName, str(slices_index) + '_' + str(slices_index+1))
                if not os.path.exists(block_save_path):
                    os.mkdir(block_save_path)
                sitk.WriteImage(sub_up, os.path.join(block_save_path, str(i)+"_"+str(j)+"up_temp_all.tif"))
                sitk.WriteImage(sub_down, os.path.join(block_save_path, str(i)+"_"+str(j)+"down_temp_all.tif"))

                # prev_surface, next_result, transform2 = CalBlock(sub_up, sub_down, spacing)
                try:
                    prev_surface, next_result, transform2 = CalBlock(sub_up,sub_down,spacing)
                    print(transform2)
                    param = transform2.GetParameters()
                    # param[2] = param[2] + drop
                    vector_points[i, j, :] = np.array(param)
                    tf_pars.append(param)
                    pos.append([i,j])
                    # todo 写入tf pars文件  这是 next img 下面那个图像的数据表面提取信息
                    with open(os.path.join(r"D:\USERS\yq\code\cal_overlap\Refine", tempName,'tf_'+str(slices_index)+'_pars.txt'), 'w') as file:
                        # 将列表的每个元素写入文件的一行
                        for k in range(len(tf_pars)):
                            file.write(str(pos[k])+": " + str(tf_pars[k]) + '\n')
                    # if tf_pars is not None:
                    #     print("tf_pars is : ",tf_pars)
                    #     break
                    print("{} {} costs time :{}".format(i,j,time.time()-start))
                    print()
                except:
                    print("row: {}; col: {} gets wrong!!!".format(i,j))
                    continue


    np.save(os.path.join(r"D:\USERS\yq\code\cal_overlap\Refine", tempName,'tf_'+str(slices_index)+'_pars.npy'),vector_points)
    print(tf_pars)
    pass
def translate_get_align_transform(fixed, moving, parameter_files, fixed_mask=None, moving_mask=None,
                        fixed_points=None, moving_points=None, rigidity_mask=None, inverse_transform=False,
                        initial_transform=None, multichannel=False):
    with tempfile.TemporaryDirectory() as ELASTIX_TEMP:
        elastix = sitk.ElastixImageFilter()
        elastix.SetOutputDirectory(ELASTIX_TEMP)
        params = sitk.VectorOfParameterMap()
        for p in parameter_files:
            param = sitk.ReadParameterFile(p)
            size = 1
            for s in moving.GetSize():
                size *= s
            if len(moving.GetSize()) == 2:
                param['NumberOfSpatialSamples'] = [str(int(max(moving.GetSize()[0] * moving.GetSize()[1] / 2048 * pow(4, i), 2048))) for i in range(4)]
            if rigidity_mask is not None:
                mask_path = os.path.join(ELASTIX_TEMP, 'rigidity_mask.mha')
                sitk.WriteImage(rigidity_mask, mask_path)
                param['MovingRigidityImageName'] = [mask_path]
            if multichannel:
                if param['Registration'][0] == 'MultiMetricMultiResolutionRegistration':
                    m = [*param['Metric']]
                    for i in range(1, fixed.GetSize()[-1]):
                        m = [*m, param['Metric'][0]]
                        param['Metric{}Weight'.format(len(m) - 1)] = param['Metric0Weight']
                    param['Metric'] = m
                m = {'FixedImagePyramid': [], 'MovingImagePyramid': [], 'Interpolator': [], 'ImageSampler': []}
                for i in range(0, len(param['Metric'])):
                    for k in m:
                        m[k] = [*m[k], param[k][0]]
                for k in m:
                    param[k] = m[k]
            params.append(param)
            #elastix.WriteParameterFile(param, 'F:/chaoyu/test/f.txt')
        elastix.SetParameterMap(params)
        if multichannel:
            for c in range(fixed.GetSize()[-1]):
                idx = *[slice(None) for i in range(len(fixed.GetSize()) - 1)], c
                elastix.AddFixedImage(fixed[idx])
                elastix.AddMovingImage(moving[idx])
            for i in range(len(param['Metric']) - fixed.GetSize()[-1]):
                idx = *[slice(None) for i in range(len(fixed.GetSize()) - 1)], 0
                elastix.AddFixedImage(fixed[idx])
                elastix.AddMovingImage(moving[idx])
        else:
            elastix.SetFixedImage(fixed)
            elastix.SetMovingImage(moving)
        if fixed_mask is not None:
            elastix.SetFixedMask(fixed_mask)
        if moving_mask is not None:
            elastix.SetMovingMask(moving_mask)
        if fixed_points is not None and moving_points is not None:
            elastix.SetFixedPointSetFileName(fixed_points)
            elastix.SetMovingPointSetFileName(moving_points)
        if initial_transform is not None:
            elastix.SetInitialTransformParameterFileName(initial_transform)
        s = elastix.Execute()
        tf_par = elastix.GetTransformParameterMap()
        tp_ = elastix.GetTransformParameterMap()[0]['TransformParameters']
        tp_ = [float(i) for i in tp_]
        return tp_