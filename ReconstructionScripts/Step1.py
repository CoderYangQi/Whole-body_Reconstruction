'''
@ yangqi
本步骤实现 offset 的校准 以及计算粗配准面

'''

'''
@ yangqi
使用修复的 3D detect surface的方法 来检测数据
block size = 250 * 250;   cal size 125 * 125 * 40

                down_temp = down_img[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size,
                          end2 - interval//2 :end2 + interval - interval//2]

'''

from VISoR_Reconstruction.reconstruction.brain_reconstruct_methods.common import fill_outside
from common_script.common0424 import *

def CalSurfaceTranslate(prev_surface_path, next_surface_path):
    def PreProcess(img):
        img = sitk.Cast(img, sitk.sitkFloat32)
        refineImg = sitk.Clamp((sitk.Log(sitk.Cast(img, sitk.sitkFloat32)) - 4.6) * 39.4, sitk.sitkUInt8, 0, 255)
        return refineImg
    translateDict = {}

    # 选取 index 为 33 的数据进行测试
    prev_surface = sitk.ReadImage(prev_surface_path)
    next_surface = sitk.ReadImage(next_surface_path)
    prev_surface = PreProcess(prev_surface)
    next_surface = PreProcess(next_surface)
    prev_size = prev_surface.GetSize()
    next_size = next_surface.GetSize()
    ref_scale = 1
    outside_brightness = 2
    # next_surface = ResizeImg(next_surface, next_size, ref_scale)
    prev_surface = fill_outside(prev_surface, outside_brightness)
    next_surface = fill_outside(next_surface, outside_brightness)
    # 此处 按照 next 为 fixed ； prev 为 moving； 因为需要计算 next的上表面偏移量
    tp_ = translate_get_align_transform(next_surface, prev_surface,
                                        [os.path.join(PARAMETER_DIR,
                                                      'yq_align_surface_2D.txt')])

    return tp_


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

# todo 根据origin 和整个 bounds 来重新 SliceImage
def SliceResample(imgPath,leftPoint, point, refSize,savePath,checklsPath, checkusPath):

    img = sitk.ReadImage(imgPath)

    imgSize = img.GetSize()
    img.SetSpacing([4,4,4])
    # sliceOrigin = pointsPair[0]
    img.SetOrigin(point)
    newSize = [refSize[0],refSize[1],imgSize[2]]
    refineImg = sitk.Resample(img,newSize,sitk.Transform(),sitk.sitkLinear,leftPoint,[4,4,4])
    # refineImg = sitk.Resample(img,img,sitk.Transform(),sitk.sitkLinear,leftPoint,[4,4,4])
    sitk.WriteImage(refineImg[:,:,175],checklsPath)
    sitk.WriteImage(refineImg[:,:,75],checkusPath)
    # write_ome_tiff(refineImg, savePath)
    pass
# todo 将 75 上下的数据进行maxprojection
def MaxProjSurface(imgPath, usSavePath, lsSavePath):
    img = sitk.ReadImage(imgPath)
    usIndex = 75
    lsIndex = 175
    maxThickness = 20
    us = sitk.MaximumProjection(img[:,:,usIndex - maxThickness//2 : usIndex + maxThickness//2],projectionDimension=2)[:,:,0]
    ls = sitk.MaximumProjection(img[:,:,lsIndex - maxThickness:lsIndex],projectionDimension=2)[:,:,0]
    sitk.WriteImage(us, usSavePath)
    sitk.WriteImage(ls, lsSavePath)


    return None




import multiprocessing
import time, gc

def run_multiprocess(numsThread, taskParas):
    # todo use multiprocess
    pool = multiprocessing.Pool(numsThread)
    result = []
    for i in range(len(taskParas)):
        msg = 'hello %s' % i
        result.append(pool.apply_async(func=taskFun, args=taskParas[i]))

    pool.close()
    pool.join()

    # for res in result:
    #     print('***:', res.get())  # get()函数得出每个返回结果的值

    print('All end--')



def taskFun(up_path, down_path, upOrigin, downOrigin, left_point, refSize, spacing, i,bottom1,end2,
            checklsPath = None, checkusPath = None):
    # 解包数据

    # 模拟重建算法的任务
    print(f"Reconstruction started for data chunk ")
    print(f"String input: {up_path, down_path}")


    # print(f"Reconstruction completed for data chunk {data_id}")
    up_img = sitk.ReadImage(up_path)
    down_img = sitk.ReadImage(down_path)

    # 统一 数据的大小范围
    # todo 默认不做 扩充，但是可能造成数据的缺失
    # left_point = [0,0,0]
    print("left_point is : ", left_point)
    # todo

    # todo 不需要做全局的 填充，仅仅只用在意邻近片之间的问题
    up_img.SetOrigin(upOrigin)
    up_img.SetSpacing(spacing)
    down_img.SetOrigin(downOrigin)
    down_img.SetSpacing(spacing)
    # init transform
    dimension = 3
    up_size = up_img.GetSize()
    up_img = sitk.Resample(up_img, [refSize[0], refSize[1], up_size[2]],
                             sitk.Transform(), sitk.sitkLinear, left_point, spacing)

    down_size = down_img.GetSize()
    down_img = sitk.Resample(down_img, [refSize[0], refSize[1], down_size[2]],
                             sitk.Transform(), sitk.sitkLinear, left_point, spacing)
    # sitk.WriteImage()

    print("down_img.GetSpacing() : {}\n Origin: {} \n Size: {}".format(down_img.GetSpacing(), down_img.GetOrigin(),
                                                                       down_img.GetSize()))
    print("up_img.GetSpacing() : {}\n Origin: {}\n Size: {}".format(up_img.GetSpacing(), up_img.GetOrigin(),
                                                                    up_img.GetSize()))
    sitk.WriteImage(up_img[:,:,75],r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\TestTemp\th2_30_36\{}_75.tif".format(i))
    sitk.WriteImage(up_img[:,:,175],r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\TestTemp\th2_30_36\{}_175.tif".format(i))
    # todo 获得 xy 的粗校准
    start = time.time()

    #  利用 最大值投影的2D data 计算位移（和旋转角度）
    size1 = up_img.GetSize()
    size2 = down_img.GetSize()
    # 计算高度 圈定大概的数据范围
    # bottom1 = GetBottom_4um(size1)
    # bottom2 = GetBottom_4um(size2)
    # end2 = int(bottom2 - 40 * 2.5)
    interval = 40
    # todo 使用 4 微米的图像进行测试
    # spacing = [4,4,4]

    next_result = None
    # tf, next_result = cal_xyz(up_img,down_img,spacing,bottom1,end2,interval)
    # tf_para = tf.GetParameters()
    # transform = sitk.TranslationTransform(3)
    # transform.SetParameters([tf_para[0], tf_para[1], 0])
    # down_img = sitk.Resample(down_img, down_img, transform, sitk.sitkLinear)
    #
    # Max_up_img = sitk.MaximumProjection(up_img, projectionDimension=2)[:, :, 0]
    # Max_down_img = sitk.MaximumProjection(down_img, projectionDimension=2)[:, :, 0]
    # sitk.WriteImage(Max_up_img, r'D:\USERS\yq\TH2_Reconstruction\' + 'MAX_' + str(up) + '.tif')
    # sitk.WriteImage(Max_down_img, r'D:\USERS\yq\TH2_Reconstruction\' + 'MAX_' + str(down) + '.tif')

    print("粗校准 花费的时间为： {}".format(time.time() - start))

    SeparateBlock(next_result, up_img, down_img, spacing, bottom1, end2, interval, slices_index=i)

    print("the space of {} cost : {} ".format(i, time.time() - start))
    gc.collect()
def ReadNPY():
    a = np.load("Refine/tf_155_pars.npy")
    a_0 = a[:,:,0]
    a_1 = a[:,:,1]
    a_2 = a[:,:,2]
    print()
def TH_Main():
    # 获取 height 替代 175 和 75
    # 读取 slicetransform 里面 文件
    init_data = Reconstruction_Point()
    init_data.init()
    init_data.create_750()
    right_point = init_data.rightbottom
    left_point = init_data.lefttop
    points = init_data.points
    pxSize = init_data.internal_pixel_size
    # spacing = [pxSize for i in range(3)]
    spacing = [4.0, 4.0, 4.0]
    channels = init_data.channels
    num = int(len(points) / 2)
    slices = init_data.slices
    rate = 1

    # SliceImageRoot = os.path.join(init_data.basePath,'SliceImage', '10.0')

    # todo 4 um 的重构数据
    # SliceImageRoot = r"Z:\Data\E\E-123\1108 Reconstruction\SliceImage\4.0"
    SliceImageRoot = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\SliceImage\4.0"
    up_path_format =r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\TestTemp\RefineSliceImage\translate_img_{:03d}.tif"
    # SliceImageRoot = os.path.join(init_data.basePath, 'SliceImage', '10.0')

    # TODO 遍历数据 然后计算粗校准数据 拿前五个出来测试
    count = 0
    slices_keys = [i for i in slices]
    slices_keys.append(len(slices))
    taskChunk = []
    for i in range(30, 36):
        slice_index = i - 1
        start = time.time()
        up = slices_keys[i] - 1  # 此处 i == 153，就是表示 image index == 154
        down = slices_keys[i + 1] - 1

        # up_path = os.path.join(SliceImageRoot, os.path.basename(slices[up].path).split('.')[0] + '.tif')
        up_path = up_path_format.format(i)
        down_path = os.path.join(SliceImageRoot, os.path.basename(slices[down].path).split('.')[0] + '.tif')
        # bottom1 = heightPairs[i][1]
        bottom1 = 175
        end2 = 75
        temp = (up_path, down_path, up, down, left_point, points, slice_index, spacing, i, bottom1, end2)
        taskChunk.append(temp)

    num_threads = 5  # 设置线程数量
    # data_chunks = [(1, "abc", 42), (2, "xyz", 18), (3, "def", 99)]  # 设置数据切片，每个元素包含多个输入类型

    # run_reconstruction_with_fixed_threads(num_threads, taskChunk)

    run_multiprocess(num_threads, taskChunk)

    # # 使用线程池
    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     # 提交任务给线程池
    #     futures = [executor.submit(taskFun, chunk) for chunk in taskChunk]
    #
    #     # 等待所有任务完成
    #     for future in futures:
    #         future.result()
    # print("All reconstruction tasks completed")

    print()



def main():
    # todo read reconstruction info and use the point bounds to resample the size of the image
    # todo get flsm orign offset info
    # visorPath = r"D:\USERS\yq\TH2_Reconstruction\30-36.visor"
    # leftList, rightList = GetOffset(visorPath)
    # leftList = np.array(leftList)
    # rightList = np.array(rightList)
    #
    # spacing = [4,4,4]
    # lefttop = leftList.min(axis=0)
    # rightbottom = rightList.max(axis=0)
    # print()
    #
    # # # todo Resample SliceImage after get offsets
    # # 计算 最大的边界数值
    # refSize = [(rightbottom[0] - lefttop[0])//spacing[0], (rightbottom[1] - lefttop[1])//spacing[1]]
    # refSize = [int(i) for i in refSize]
    # print(f"refSize is {refSize}")
    # saveFormat = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\TestTemp\RefineSliceImage\0510resize_{:03d}_img.tif"
    # imgFormat = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\SliceImage\4.0\2_1_1_{:03d}_561nm_10X.tif"
    # checklsFormat = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\TestTemp\RefineSliceImage\0510check_ls_{:03d}_img.tif"
    # checkusFormat = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\TestTemp\RefineSliceImage\0510check_us_{:03d}_img.tif"
    # # startOrigin = np.array([points[2 * i + 1] for i in range(185)])
    # # x = startOrigin[:,0];y = startOrigin[:,1]; z = startOrigin[:,2]
    # lefttop = [lefttop[0], lefttop[1], 0]
    # for sliceIndex in range(30,36):
    #     # imgPath,leftPoint, pointsPair, refSize,savePath  ### points 从 0 开始计算  sliceIndex 从1开始计算
    #     point = leftList[sliceIndex - 30]
    #     point = [point[0], point[1], 0]
    #     SliceResample(imgFormat.format(sliceIndex),lefttop, point,refSize
    #                   , saveFormat.format(sliceIndex),checklsFormat.format(sliceIndex),checkusFormat.format(sliceIndex))

    SliceImageRoot = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\SliceImage\4.0"
    imgFormat = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\SliceImage\4.0\2_1_1_{:03d}_561nm_10X.tif"
    up_path_format = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\TestTemp\RefineSliceImage\translate_img_{:03d}.tif"
    # SliceImageRoot = os.path.join(init_data.basePath, 'SliceImage', '10.0')

    # TODO 遍历数据 然后计算粗校准数据 拿前五个出来测试
    taskChunk = []
    visorPath = r"D:\USERS\yq\TH2_Reconstruction\delete145New.visor"
    leftList, rightList = GetOffset(visorPath)
    leftList = np.array(leftList)
    rightList = np.array(rightList)

    spacing = [4,4,4]
    lefttop = leftList.min(axis=0)
    rightbottom = rightList.max(axis=0)
    lefttop = [lefttop[0], lefttop[1], 0]
    refSize = [(rightbottom[0] - lefttop[0]) // spacing[0], (rightbottom[1] - lefttop[1]) // spacing[1]]
    refSize = [int(i) for i in refSize]
    for i in range(110, 125):
        prevIndex = i
        nextIndex = i + 1
        upOrigin = leftList[prevIndex - 1]
        upOrigin[2] = 0
        downOrigin = leftList[nextIndex - 1]
        downOrigin[2] = 0
        up_path = imgFormat.format(prevIndex)
        down_path = imgFormat.format(nextIndex)
        # bottom1 = heightPairs[i][1]
        bottom1 = 175
        end2 = 75
        temp = (up_path, down_path, upOrigin, downOrigin, lefttop, refSize, spacing, i,bottom1,end2)
        taskChunk.append(temp)

    num_threads = 7  # 设置线程数量
    # data_chunks = [(1, "abc", 42), (2, "xyz", 18), (3, "def", 99)]  # 设置数据切片，每个元素包含多个输入类型

    # run_reconstruction_with_fixed_threads(num_threads, taskChunk)

    run_multiprocess(num_threads, taskChunk)

if __name__ == '__main__':
    main()