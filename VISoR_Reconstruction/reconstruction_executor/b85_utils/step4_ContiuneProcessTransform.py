import os.path
import sys,os
sys.path.append(os.path.abspath("."))
import unittest
# from .common import Reconstruction_Point, create_folder
from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
from VISoR_Brain.utils.elastix_files import *
from VISoR_Reconstruction.reconstruction.brain_reconstruct_methods.common import fill_outside
import time
# from .torch_losses import NCC

def create_folder(name):
    if not os.path.exists(name):
        os.mkdir(name)


def copy_extract_surface(img: sitk.Image, umap: sitk.Image, lmap: sitk.Image):
    img.SetSpacing([1, 1, 1])
    img.SetOrigin([0, 0, 0])
    umap.SetSpacing([1, 1, 1])
    umap.SetOrigin([0, 0, 0])
    lmap.SetSpacing([1, 1, 1])
    lmap.SetOrigin([0, 0, 0])

    # umap_s = umap + 1
    # lmap_s = lmap - 1
    # zeros = sitk.Image(umap.GetSize(), umap.GetPixelIDValue())
    # df = sitk.JoinSeries(sitk.Compose(zeros, zeros, umap_s), sitk.Compose(zeros, zeros, lmap_s))

    df = sitk.JoinSeries(umap, lmap)

    df = sitk.Cast(df, sitk.sitkVectorFloat64)
    ref = sitk.Image(df)
    tr = sitk.DisplacementFieldTransform(3)
    tr.SetDisplacementField(df)
    surfaces = sitk.Resample(img, ref, tr)
    # surfaces = sitk.Cast(surfaces, sitk.sitkFloat32)
    # surfaces = sitk.Clamp((sitk.Log(sitk.Cast(surfaces, sitk.sitkFloat32)) - 4.6) * 39.4, sitk.sitkUInt8, 0, 255)
    # surfaces = sitk.Cast(surfaces, sitk.sitkFloat32)
    return surfaces



def CreateProcessInput(s,e,temp_root,name_format,uzlzRoot):
    # root = r"G:\Wholebody\BED\Temp_0319\temp_mha"
    # uzlzRoot = r"G:\Wholebody\BED\Temp_0319\temp_mha"

    input = {}
    output = []
    for index in range(s,e):
        # index,xy,u
        key1 = str(index)+',xy'+',u'
        input[key1] = os.path.join(temp_root, name_format + "_uxy.mha").format(index)
        # index,xy,l
        key2 = str(index) + ',xy' + ',l'
        input[key2] = os.path.join(temp_root, name_format + "_lxy.mha").format(index)
        # index,z,u
        key3 = str(index)+',z'+',u'
        input[key3] = os.path.join(uzlzRoot,name_format + "_uz.mha").format(index)
        # index,z,l
        key4 = str(index) + ',z' + ',l'
        input[key4] = os.path.join(uzlzRoot, name_format + "_lz.mha").format(index)

        # create udf ldf
        output.append(os.path.join(temp_root,name_format + "_udf.mha").format(index))
        output.append(os.path.join(temp_root,name_format + "_ldf.mha").format(index))
    # 检查所有输入文件是否存�?
    all_exist = True
    for key, file_path in input.items():
        if not os.path.exists(file_path):
            print(f"错误：文件不存在: {file_path} (key: {key})")
            all_exist = False

    if not all_exist:
        raise FileNotFoundError("部分输入文件不存在，请检查路径和文件名格式�?)

    return input,output
def ROI_ProcessTranform(s,e,temp_root,name_format,uzlzRoot):
    input, output = CreateProcessInput(s,e,temp_root,name_format,uzlzRoot)

    param = {'internal_downsample': 32, 'nonrigid': True}
    # param = {'internal_downsample': 32, 'nonrigid': False}

    from VISoR_Reconstruction.reconstruction.yq_reconstruct import roi_process_transforms_
    tf = roi_process_transforms_(input,**param)
    for i in range(len(output)):
        print(f"{i} / {len(output)}")
        sitk.WriteImage(tf[i], output[i])
def create_brain():

    # create Input
    input = {}

    create_folder(os.path.join(recon_root,'BrainTransform'))
    output = os.path.join(recon_root,'BrainTransform','visor_brain.txt')
    param = {'internal_pixel_size': 4.0, 'slice_thickness': 400}

    udfFormat = os.path.join(temp_root, name_format + "_udf.mha")
    ldfFormat = os.path.join(temp_root, name_format + "_ldf.mha")
    for index in range(s,e):
        # index,sl

        input[str(index) + ',sl'] = {'type':'reconstructed_slice','path':stFormat.format(index)}
        # index,u
        input[str(index) + ',u'] = {'type': 'image', 'path': udfFormat.format(index)}
        # index,l
        input[str(index) + ',l'] = {'type': 'image', 'path': ldfFormat.format(index)}

    input_ = {}
    for k, v in input.items():
        if v['type'] == 'image':
            # input_[k] = sitk.ReadImage(v['path'])
            new_path = v['path']
            # new_path = new_path.replace(old_part, new_part)
            input_[k] = sitk.ReadImage(new_path)
        else:
            input_[k] = VISoRSample()
            input_[k].load(v['path'])

    br = zero_create_brain_(input_, **param, output_path=output)
    br.save(output)

def refine_create_brain(s,e, temp_root, name_format,leftList, lefttop, recon_root,stFormat,start_ind):


    slice_offset_list = []
    for left in leftList:
        left = left - lefttop
        slice_offset_list.append([left[0],left[1],0])
    ct = 0
    ##
    # todo �?6~16 测试
    ct = 0
    output = os.path.join(recon_root, 'BrainTransform', 'visor_brain.txt')
    param = {'internal_pixel_size': 4.0, 'slice_thickness': 400}
    param['slice_offset_list'] = slice_offset_list
    param['start_ind'] = start_ind


    udfFormat = os.path.join(temp_root, name_format + "_udf.mha")
    ldfFormat = os.path.join(temp_root, name_format + "_ldf.mha")

    # create Input
    input = {}
    for index in range(s, e):
        # index,sl

        input[str(index) + ',sl'] = {'type': 'reconstructed_slice', 'path': stFormat.format(index)}
        # index,u
        input[str(index) + ',u'] = {'type': 'image', 'path': udfFormat.format(index)}
        # index,l
        input[str(index) + ',l'] = {'type': 'image', 'path': ldfFormat.format(index)}

    input_ = {}
    for k, v in input.items():

        if os.path.exists(v['path']):
            print(f"load create {v['path']}")
        if v['type'] == 'image':
            # input_[k] = sitk.ReadImage(v['path'])
            new_path = v['path']

            # new_path = new_path.replace(old_part, new_part)
            input_[k] = sitk.ReadImage(new_path)
        else:
            input_[k] = VISoRSample()
            input_[k].load(v['path'])

    br = refine_create_brain_(input_, **param, output_path=output)
    br.save(output)
def refine_create_brain_(input_, internal_pixel_size, slice_thickness,slice_offset_list,start_ind ,output_path=None):
    brain = VISoRBrain()

    slices = {}
    ud = {}
    ld = {}
    for k in input_:
        i = int(k.split(',')[0])
        a = k.split(',')[1]
        if a == 'sl':
            slices[i] = input_[k]
        elif a == 'u':
            ud[i] = input_[k]
        elif a == 'l':
            ld[i] = input_[k]
        input_[k] = None

    # xy_list = [[0, 0, 0], [0, 0, 0]]
    # xy_list_185 = [[0, 0, 0], [0, 0, 0]]
    # xy_list_186 = [[0, 0, 0], [320, 0, 0]]
    # xy_list_187 = [[320, 0, 0], [0, 0, 0]]
    # offset_dict = {
    #     '184': xy_list,
    #     '185': xy_list_185,
    #     '186': xy_list_186,
    #     '187': xy_list_187,
    #     '188': xy_list,
    #     '189': xy_list,
    # }
    offset_dict = {}

    for i in ud:
        sl = slices[i]
        u = ud[i]
        l = ld[i]
        if str(i) in offset_dict:
            xy_offset = offset_dict[str(i)]
        else:
            xy_offset = [[0,0,0],[0,0,0]]
        print(f"index {i} offset is {xy_offset}")

        # �?sliceindex �?1
        slice_offset = slice_offset_list[i - start_ind]
        u = sitk.Compose(sitk.VectorIndexSelectionCast(u, 0) * internal_pixel_size - slice_offset[0] + xy_offset[0][0],
                         sitk.VectorIndexSelectionCast(u, 1) * internal_pixel_size - slice_offset[1] + xy_offset[0][1],
                         sitk.VectorIndexSelectionCast(u, 2) * internal_pixel_size + (- (i - 1) * slice_thickness))
        l = sitk.Compose(sitk.VectorIndexSelectionCast(l, 0) * internal_pixel_size - slice_offset[0] + xy_offset[1][0],
                         sitk.VectorIndexSelectionCast(l, 1) * internal_pixel_size - slice_offset[1] + xy_offset[1][1],
                         sitk.VectorIndexSelectionCast(l, 2) * internal_pixel_size + (- i * slice_thickness))
        df = sitk.JoinSeries([u[:,:,0], l[:,:,0]])
        df.SetOrigin([0, 0, (i - 1) * slice_thickness])
        df.SetSpacing([internal_pixel_size, internal_pixel_size, slice_thickness])
        size = df.GetSize()


        df = sitk.Cast(df, sitk.sitkVectorFloat64)
        df = sitk.DisplacementFieldTransform(df)
        brain.slices[i] = sl
        brain.set_transform(i, df)
        brain.slice_spheres[i] = [[0, 0, (i - 1) * slice_thickness],
                                  [size[0] * internal_pixel_size, size[1] * internal_pixel_size, i * slice_thickness]]
        if output_path is not None:
            brain.save(output_path)
            brain.release_transform(i)
        ud[i] = None
        ld[i] = None
    brain.calculate_sphere()
    return brain

def GetOffset(visorPath):
    def Read(file_path):
        # Replace 'path_to_your_file.txt' with the actual path to your text file
        try:
            # Open the text file and read its contents
            with open(file_path, 'r') as file:
                data = file.read()
                # Parse the data as JSON
                json_data = json.loads(data)

                # Extract the specific values
                lefttop_x = json_data['lefttop_x']
                lefttop_y = json_data['lefttop_y']
                lefttop_z = json_data['lefttop_z']

                # Extract the specific values
                rightbottom_x = json_data['rightbottom_x']
                rightbottom_y = json_data['rightbottom_y']
                rightbottom_z = json_data['rightbottom_z']

                # Print the extracted values
                print("lefttop_x:", lefttop_x)
                print("lefttop_y:", lefttop_y)
                print("lefttop_z:", lefttop_z)
                left = [eval(lefttop_x) * 1e3, eval(lefttop_y) * 1e3, float(lefttop_z) * 1e3]
                right = [eval(rightbottom_x) * 1e3, eval(rightbottom_y) * 1e3, float(rightbottom_z) * 1e3]
            return left, right
        except FileNotFoundError:
            print("The file was not found. Please check the path.")
        except json.JSONDecodeError:
            print("Failed to decode JSON. Please check the file content.")
        except KeyError:
            print("One or more keys were not found in the JSON data.")

    # todo get all flsm files
    with open(visorPath) as f:
        info = json.load(f)
    directory_path = os.path.dirname(visorPath)
    acquisition = info['Acquisition Results']
    pathList = []
    leftList = []
    rightList = []
    for flstDict in acquisition:
        temp = flstDict['FlsmList'][0]
        path = os.path.join(directory_path, temp)
        pathList.append(path)
        sliceID = flstDict['SliceID']
        left,right = Read(path)
        leftList.append(left)
        rightList.append(right)
    return leftList , rightList

def generate_brain_image(brainPath, imgPath, slice_index, input_pixel_size, output_pixel_size, name_format, n_start,
                         roi=None, slice_origin=None, bit_downsample=True):
    brain = VISoRBrain()
    brain.load(brainPath)
    img = sitk.ReadImage(imgPath)
    if slice_origin is None:
        # slice_origin = brain.slices[slice_index].sphere[0]
        # # todo 可能是数据有问题
        slice_origin = [0,0,0]
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
def _generate_brain_image(s,e,recon_root,imgFormat,leftList,start_ind):
    # br,imgPath,i,name_format,n_start = taskpara
    create_folder(os.path.join(recon_root,'BrainImage'))
    create_folder(os.path.join(recon_root, 'BrainImage','4.0'))
    name_format = os.path.join(os.path.join(recon_root, 'BrainImage','4.0'), 'Z{:05d}_' + 'C{}.tif'.format('1'))
    br = VISoRBrain()
    brainPath = os.path.join(recon_root,'BrainTransform','visor_brain.txt')

    # br.load(brainPath)
    taskChunks = []
    for sliceIndex in range(s, e):
        imgPath = imgFormat.format(sliceIndex)
        n_start = 100 * (sliceIndex - 1)
        slice_index = sliceIndex;
        input_pixel_size = 4.0;
        output_pixel_size = 4.0
        originIndex = sliceIndex - start_ind # 自动 -1
        imgOrigin = leftList[originIndex]
        imgOrigin = [imgOrigin[0], imgOrigin[1], 0]
        temp = (brainPath, imgPath, slice_index, input_pixel_size, output_pixel_size, name_format, n_start)
        taskChunks.append(temp)
        # yq_generate_brain_image(brainPath, imgPath, slice_index, input_pixel_size, output_pixel_size
        #         , name_format, n_start,imgOrigin,refSize,lefttop)

    num_threads = 1  # 设置线程数量
    brainimage_multiprocess(num_threads, taskChunks)
    # print(imgPath,sliceIndex,n_start)
    # for sliceIndex in range(131,170):
    #     imgPath = imgFormat.format(sliceIndex)
    #     n_start = 100 * (sliceIndex - 1)
    #     img = sitk.ReadImage(imgPath)
    #     # img = RefineImg(img,refSize,imgOrigin, lefttop)
    #     t_dummy = yq_generate_brain_image(brain=br, img = img, slice_index=sliceIndex
    #                                    , input_pixel_size=1.0,output_pixel_size=1.0,name_format=name_format,
    #                                    n_start=n_start)
import multiprocessing
import time, gc
def brainimage_multiprocess(numsThread, taskParas):
    # todo use multiprocess
    pool = multiprocessing.Pool(numsThread)
    result = []
    for i in range(len(taskParas)):
        msg = 'hello %s' % i
        result.append(pool.apply_async(func=generate_brain_image, args=taskParas[i]))

    pool.close()
    pool.join()
def yq_generate_brain_image(brainPath, imgPath, slice_index, input_pixel_size, output_pixel_size, name_format, n_start,
                        imgOrigin,refSize,lefttop
                         ,roi=None, slice_origin=None, bit_downsample=True):
    brain = VISoRBrain()
    brain.load(brainPath)
    if slice_origin is None:
        slice_origin = brain.slices[slice_index].sphere[0]
        # # todo 可能是数据有问题
        # slice_origin[2] = 0
    img = sitk.ReadImage(imgPath)


    nextSize = img.GetSize()
    spacing = [4, 4, 4]
    # img_size = [nextSize[0], nextSize[1]]
    # todo 对图像进�?Resample 和之前的计算粗校准面的坐标一�?
    img.SetOrigin(imgOrigin)
    img.SetSpacing(spacing)
    img = sitk.Resample(img, [refSize[0], refSize[1], nextSize[2]], sitk.Transform(), sitk.sitkLinear, lefttop,
                        [4, 4, 4])

    img.SetOrigin([0, 0, 0])
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

def CheckInput(s,e,temp_root, name_format,uzlzRoot):
    # root = r"G:\Wholebody\BED\Temp_0319\temp_mha"
    # uzlzRoot = r"G:\Wholebody\BED\Temp_0319\temp_mha"
    # s, e = 1, 177
    # #  ROI_ProcessTranform() init()
    # # temp_root = r"K:\STZ1_914#\save_temp_0313\Temp"
    # temp_root = r"I:\STZ914\Temp_Large"
    # uzlzRoot = r"I:\STZ914\Temp_Large"
    # name_format = "DBH_CHAT-TH_1_{:03d}_648nm_10X"

    input = {}
    output = []
    for index in range(s,e):
        # index,xy,u
        key1 = str(index)+',xy'+',u'
        input[key1] = os.path.join(temp_root, name_format + "_uxy.mha").format(index)
        # index,xy,l
        key2 = str(index) + ',xy' + ',l'
        input[key2] = os.path.join(temp_root, name_format + "_lxy.mha").format(index)
        # index,z,u
        key3 = str(index)+',z'+',u'
        input[key3] = os.path.join(uzlzRoot,name_format + "_uz.mha").format(index)
        # index,z,l
        key4 = str(index) + ',z' + ',l'
        input[key4] = os.path.join(uzlzRoot, name_format + "_lz.mha").format(index)

        # create udf ldf
        # output.append(os.path.join(temp_root,name_format + "_udf.mha").format(index))
        # output.append(os.path.join(temp_root,name_format + "_ldf.mha").format(index))
    # 检查所有输入文件是否存�?
    all_exist = True
    for key, file_path in input.items():
        if not os.path.exists(file_path):
            print(f"错误：文件不存在: {file_path} (key: {key})")
            all_exist = False
    return all_exist

if __name__ == '__main__':
    # main()
    # RenameMha()
    # CreateProcessInput()
    # ROI_ProcessTranform()
    # create_brain()
    # _generate_brain_image()

    # readmha()
    # mainCreateXY()

    # extract_surface()

    # 获取了uxy lxy �?-�?get udf ldf
    # CreateProcessInput() # 修改 路径

    # check_interval = 1 * 6  # 检查间隔：10分钟�?00秒）
    #
    # # while True:
    # #     print("开始检查文件是否就绪…�?)
    # #     if CheckInput():
    # #         print("检查完毕：所有文件均已就绪�?)
    # #         break
    # #     else:
    # #         print(f"检查结果：部分文件未就绪。{check_interval / 60} 分钟后再次检查�?)
    # #         time.sleep(check_interval)
    #
    # s, e = 36,42
    # # s2, e2 = 53,61
    # #  ROI_ProcessTranform() init()
    # # temp_root = r"K:\STZ1_914#\save_temp_0313\Temp"
    # temp_root = r"E:\20250426_SMY_TAC1_AI14_1_1\Reconstruction\tac\temp_block"
    # uzlzRoot = r"E:\20250426_SMY_TAC1_AI14_1_1\Reconstruction\tac\temp_block"
    # name_format = "TAC1_AI14_1_{:03d}_561nm_10X"
    # CheckInput()
    # # temp_root = r"K:\STZ1_914#\save_temp_0313\Temp"
    # ori_temp_root = r"E:\20250426_SMY_TAC1_AI14_1_1\Reconstruction\tac"
    # recon_root = os.path.join(ori_temp_root, "BrainTrans_{}_{}_0604".format(s, e - 1))
    # os.makedirs(recon_root, exist_ok=True)
    # stFormat = r"E:\Reconstruction\SliceTransform\\" + name_format + ".txt"
    #
    # # init generate brain image
    # tif_name_format = name_format
    # imgFormat = r"E:\Reconstruction\SliceImage\4.0\\" + tif_name_format + ".tif"
    # visorPath = "E:\\tac1_ai14.visor"
    # leftList, rightList = GetOffset(visorPath)
    # leftList = np.array(leftList)
    # rightList = np.array(rightList)
    #
    # spacing = [4, 4, 4]
    # lefttop = leftList.min(axis=0)
    # rightbottom = rightList.max(axis=0)
    # lefttop = [lefttop[0], lefttop[1], 0]
    # refSize = [10500,6750]
    # 
    # CheckInput()
    #
    # ROI_ProcessTranform()
    #
    # # create_brain()
    # refine_create_brain()
    #
    # _generate_brain_image()
    #
    #
    #
    # # img_calculate_height()
    # # _648nm_generate_brain_image()

    print()

