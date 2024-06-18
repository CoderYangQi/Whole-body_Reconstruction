import argparse
import json
import re
import unittest
import os
import SimpleITK as sitk
from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
def mainCreateXY():
    root = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\Temp"
    uxyFormat = os.path.join(root,r"2_1_1_{:03d}_561nm_10X_uxy.mha")
    lxyFormat = os.path.join(root,r"2_1_1_{:03d}_561nm_10X_lxy.mha")
    uzFormat = os.path.join(root,r"2_1_1_{:03d}_561nm_10X_uz.mha")
    lzFormat = os.path.join(root,r"2_1_1_{:03d}_561nm_10X_lz.mha")
    img_size =  [8000,7200]
    for sliceIndex in range(101,102):
        print(f"sliceIndex is {sliceIndex}")
        uxy = sitk.Image()

        umap_z = sitk.Image([img_size[0], img_size[1]], sitk.sitkFloat32) + 75

        umap_y = sitk.Image([img_size[0], img_size[1]], sitk.sitkFloat32);
        umap_x = sitk.Image([img_size[0], img_size[1]], sitk.sitkFloat32)

        lmap_z = sitk.Image([img_size[0], img_size[1]], sitk.sitkFloat32) + 175

        lmap_y = sitk.Image([img_size[0], img_size[1]], sitk.sitkFloat32);
        lmap_x = sitk.Image([img_size[0], img_size[1]], sitk.sitkFloat32)
        # todo  使用 zero 替代 x和y的位移

        # todo 保存 uz lz mha 不同于 之前的 uz lz 此处的数据是有 x y z 三个维度的形变场

        uz = sitk.Compose(umap_x, umap_y, umap_z)
        lz = sitk.Compose(lmap_x, lmap_y, lmap_z)
        uxy = sitk.Compose(umap_x, umap_y)
        lxy = sitk.Compose(lmap_x,lmap_y)
        sitk.WriteImage(uxy,uxyFormat.format(sliceIndex))
        sitk.WriteImage(lxy, lxyFormat.format(sliceIndex))

    pass
# todo 将 位移场 添加offset到BrainTransform中
def create_brain():
    visorPath = r"D:\USERS\yq\TH2_Reconstruction\delete145New.visor"
    leftList, rightList = ReadVISoR(visorPath)
    leftList = np.array(leftList)
    rightList = np.array(rightList)
    lefttop = leftList.min(axis=0)
    lefttop = [lefttop[0], lefttop[1], 0]
    rightbottom = rightList.max(axis=0)
    slice_offset_list = []
    for left in leftList:
        left = left - lefttop
        slice_offset_list.append([left[0],left[1],0])
    ct = 0
    ##
    with open(r'D:\USERS\yq\TH2_Reconstruction\Reconstruction/ReconstructionInput.json') as f:
        doc = json.load(f)['tasks']['create_brain_2_1_1']
    input_ = {}
    param = doc['parameters']

    param['slice_offset_list'] = slice_offset_list
    output = doc['output_targets'][0]['path']
    # todo 取 6~16 测试
    ct = 0
    for k, v in doc['input_targets'].items():
        # if ct < 30:
        #     ct += 1;
        #     continue
        if v['type'] == 'image':
            input_[k] = sitk.ReadImage(v['path'])
        else:
            input_[k] = VISoRSample()
            input_[k].load(v['path'])
        # ct += 1
        # # 取 30 个 停止
        # if ct == 45:
        #     break
    # for k, v in doc['input_targets'].items():
    #     if v['type'] == 'image':
    #         input_[k] = sitk.ReadImage(v['path'])
    #     else:
    #         input_[k] = VISoRSample()
    #         input_[k].load(v['path'])

    br = yq_create_brain_(input_, **param, output_path=output)
    br.save(output)
def ApplyBrainTransform():
    internal_pixel_size = 4.0
    slice_thickness = 400
    visorPath = r"D:\USERS\yq\TH2_Reconstruction\delete145New.visor"
    leftList, rightList = ReadVISoR(visorPath)
    leftList = np.array(leftList)
    rightList = np.array(rightList)
    lefttop = leftList.min(axis=0)
    lefttop = [lefttop[0], lefttop[1], 0]
    rightbottom = rightList.max(axis=0)
    slice_offset_list = []
    for left in leftList:
        left = left - lefttop
        slice_offset_list.append([left[0], left[1], 0])
    # read df
    udfformat = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\Temp\2_1_1_{:03d}_561nm_10X_udf.mha"
    ldfformat = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\Temp\2_1_1_{:03d}_561nm_10X_ldf.mha"
    imgFormat = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\SliceImage\4.0\2_1_1_{:03d}_561nm_10X.tif"
    name_format = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\TestTemp\th2_0603\Z{:05d}_C1.tif"
    saveRoot = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\TestTemp\th2_0603"
    index = 112
    n_start = 4 * (index - 1)
    bit_downsample = True
    output_pixel_size = 4.0
    i = index
    u = sitk.ReadImage(udfformat.format(index))
    l = sitk.ReadImage(ldfformat.format(index))
    img = sitk.ReadImage(imgFormat.format(index))
    img.SetSpacing([4,4,4])
    # apply
    slice_offset = slice_offset_list[i - 1]
    u = sitk.Compose(sitk.VectorIndexSelectionCast(u, 0) * internal_pixel_size - slice_offset[0],
                     sitk.VectorIndexSelectionCast(u, 1) * internal_pixel_size - slice_offset[1],
                     sitk.VectorIndexSelectionCast(u, 2) * internal_pixel_size + (- (i - 1) * slice_thickness))
    l = sitk.Compose(sitk.VectorIndexSelectionCast(l, 0) * internal_pixel_size - slice_offset[0],
                     sitk.VectorIndexSelectionCast(l, 1) * internal_pixel_size - slice_offset[1],
                     sitk.VectorIndexSelectionCast(l, 2) * internal_pixel_size + (- i * slice_thickness))
    df = sitk.JoinSeries([u[:, :, 0], l[:, :, 0]])
    df.SetOrigin([0, 0, (i - 1) * slice_thickness])
    df.SetSpacing([internal_pixel_size, internal_pixel_size, slice_thickness])
    size = df.GetSize()
    df = sitk.Cast(df, sitk.sitkVectorFloat64)
    df = sitk.DisplacementFieldTransform(df)

    roi = [[0, 0, (i - 1) * slice_thickness],
     [size[0] * internal_pixel_size, size[1] * internal_pixel_size, i * slice_thickness]]

    # apply generate brain image
    size = [int((roi[1][j] - roi[0][j]) / output_pixel_size)
            for j in range(3)]
    print(size)
    res = sitk.Resample(img, size, df, sitk.sitkLinear, roi[0],
                        [output_pixel_size, output_pixel_size, output_pixel_size])
    sitk.WriteImage(res[:,:,0],os.path.join(saveRoot,"{}_0.tif".format(index)))
    res.SetSpacing([j / 1000 for j in res.GetSpacing()])
    paths = [name_format.format(n_start + j) for j in range(size[2])]
    if not os.path.exists(os.path.dirname(paths[0])):
        os.makedirs(os.path.dirname(paths[0]))
    for i in range(size[2]):
        m = sitk.GetArrayFromImage(res[:, :, i])
        if bit_downsample:
            m = np.left_shift(np.right_shift((m + 8), 4), 4)
        tifffile.imwrite(paths[i], m, compress=1)
def generate_brain_image(brain: VISoRBrain, img, slice_index, input_pixel_size, output_pixel_size, name_format, n_start,
                         roi=None, slice_origin=None, bit_downsample=True):
    if slice_origin is None:
        slice_origin = brain.slices[slice_index].sphere[0]
        # # todo 可能是数据有问题
        # slice_origin[2] = 0
    img.SetOrigin(slice_origin)
    img.SetSpacing([input_pixel_size, input_pixel_size, input_pixel_size])
    if roi is None:
        roi = brain.slice_spheres[slice_index]
    size = [int((roi[1][j] - roi[0][j]) / output_pixel_size)
            for j in range(3)]
    print(size)
    res = sitk.Resample(img, size, brain.transform(slice_index), sitk.sitkLinear, roi[0],
                        [output_pixel_size, output_pixel_size, output_pixel_size])




    res.SetSpacing([j / 1000 for j in res.GetSpacing()])
    paths = [name_format.format(n_start + j) for j in range(size[2])]
    if not os.path.exists(os.path.dirname(paths[0])):
        os.makedirs(os.path.dirname(paths[0]))
    for i in range(size[2]):
        m = sitk.GetArrayFromImage(res[:, :, i])
        if bit_downsample:
            m = np.left_shift(np.right_shift((m + 8), 4), 4)
        tifffile.imwrite(paths[i], m, compress=1)
    file_list = paths.__str__()[2:-2].replace('\', \'', '\n')
    return file_list

if __name__ == '__main__':
    # mainCreateXY()
    # process_transform()
    # create_brain()
    ApplyBrainTransform()
    # _generate_brain_image()