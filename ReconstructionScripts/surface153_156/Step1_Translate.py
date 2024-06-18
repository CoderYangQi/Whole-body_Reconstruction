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
    使用roi数据，加强数据检测的block密度

'''

from VISoR_Reconstruction.reconstruction.brain_reconstruct_methods.common import fill_outside
from common_script.common0604 import *

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



def taskFun(up_path, down_path, upOrigin, downOrigin, left_point, refSize, spacing, i,bottom1,end2,saveRoot,
            checklsPath = None, checkusPath = None):
    # # 解包数据
    # img_roi = [[7000,8900],[4300,5920]]
    # 模拟重建算法的任务
    print(f"Reconstruction started for data chunk ")
    print(f"String input: {up_path, down_path}")


    # print(f"Reconstruction completed for data chunk {data_id}")
    up_img = sitk.ReadImage(up_path)
    down_img = sitk.ReadImage(down_path)

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

    # 统一 数据的大小范围
    # todo 默认不做 扩充，但是可能造成数据的缺失
    # left_point = [0,0,0]
    print("left_point is : ", left_point)
    # todo


    # sitk.WriteImage()

    print("down_img.GetSpacing() : {}\n Origin: {} \n Size: {}".format(down_img.GetSpacing(), down_img.GetOrigin(),
                                                                       down_img.GetSize()))
    print("up_img.GetSpacing() : {}\n Origin: {}\n Size: {}".format(up_img.GetSpacing(), up_img.GetOrigin(),
                                                                    up_img.GetSize()))
    # sitk.WriteImage(up_img[:,:,80],r"D:\USERS\yq\TH2_Reconstruction\ROI_130_151\surface\{}_700.tif".format(i))
    # sitk.WriteImage(down_img[:,:,80],r"D:\USERS\yq\TH2_Reconstruction\ROI_130_151\surface\{}_300.tif".format(i + 1))
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
    roi = [[140,180], [40,80]]
    # todo 使用 4 微米的图像进行测试
    # spacing = [4,4,4]


    next_result = None
    print("粗校准 花费的时间为： {}".format(time.time() - start))

    SeparateBlock(next_result, up_img, down_img, spacing, bottom1, end2,saveRoot = saveRoot,block_size = 500,sub_block = 500, roi=roi, slices_index=i,tempName = 'th0630_153_155')

    print("the space of {} cost : {} ".format(i, time.time() - start))
    gc.collect()
def ReadNPY():
    a = np.load("Refine/tf_155_pars.npy")
    a_0 = a[:,:,0]
    a_1 = a[:,:,1]
    a_2 = a[:,:,2]
    print()
def main():
    # todo read reconstruction info and use the point bounds to resample the size of the image
    # imgFormat = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\new\130_620_720.tif"
    prevImgFormat = r"Z:\Data\E\E-123\Reconstruction\SliceImage\{}_img.tif"
    nextImgFormat = r"Z:\Data\E\E-123\Reconstruction\SliceImage\{}_img.tif"
    saveRoot = r"Z:\Data\E\E-123\Reconstruction\saveTemp"
    up_path_format = r"D:\USERS\yq\TH2_Reconstruction\Reconstruction\TestTemp\RefineSliceImage\translate_img_{:03d}.tif"
    # SliceImageRoot = os.path.join(init_data.basePath, 'SliceImage', '10.0')

    # TODO 遍历数据 然后计算粗校准数据 拿前五个出来测试
    taskChunk = []
    visorPath = r"Z:\Data\E\E-123\0630test_2.visor"
    leftList, rightList = GetOffset(visorPath)
    leftList = np.array(leftList)
    rightList = np.array(rightList)

    spacing = [1,1,1]
    newLeftList = leftList[152:156]
    newRightList = rightList[152:156]
    lefttop = newLeftList.min(axis=0)
    rightbottom = newRightList.max(axis=0)
    lefttop = [lefttop[0], lefttop[1], 0]

    rate = 4
    refSize = [(rightbottom[0] - lefttop[0]) // rate, (rightbottom[1] - lefttop[1]) // rate]
    refSize = [int(i) for i in refSize]
    print(f"refine size: {refSize}")
    lefttop2 = [0, 0, 0]
    for i in range(153, 156):
        prevIndex = i
        nextIndex = i + 1
        # upOrigin = leftList[prevIndex - 1]
        upOrigin = [0,0,0]
        upOrigin[2] = 0
        # downOrigin = leftList[nextIndex - 1]
        downOrigin = [0,0,0]
        downOrigin[2] = 0
        up_path = prevImgFormat.format(prevIndex)
        down_path = nextImgFormat.format(nextIndex)
        # bottom1 = heightPairs[i][1]
        bottom1 = 175
        end2 = 75
        temp = (up_path, down_path, upOrigin, downOrigin, lefttop2, refSize, spacing, i,bottom1,end2,saveRoot)
        taskChunk.append(temp)

    num_threads = 3  # 设置线程数量
    # data_chunks = [(1, "abc", 42), (2, "xyz", 18), (3, "def", 99)]  # 设置数据切片，每个元素包含多个输入类型

    # run_reconstruction_with_fixed_threads(num_threads, taskChunk)

    run_multiprocess(num_threads, taskChunk)

if __name__ == '__main__':
    main()