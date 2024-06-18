from VISoR_Reconstruction.reconstruction.yq_reconstruct import *
from VISoR_Brain.utils.elastix_files import *


def main():
    root = r"D:\USERS\yq\TH2_Part"
    TaskList = ["1_30","30_60","60_90","90_110", "110_125","125_155"]
    for i in range(2,4):
        fileName = os.path.join(root,TaskList[i],"Reconstruction/ReconstructionInput.json")
        with open(fileName) as f:
            doc = json.load(f)['tasks']['process_transforms_2_1_1']
        test_temp = {i: (doc['input_targets'][i]['path']) for i in doc['input_targets']}
        output_image = [i['path'] for i in doc['output_targets']]
        param = doc['parameters']
        param['nonrigid'] = True
        print(f"start {fileName}")
        input_image = {i: sitk.ReadImage(doc['input_targets'][i]['path']) for i in doc['input_targets']}

        tf = process_transforms_(input_image, **param)
        output_image = [i['path'] for i in doc['output_targets']]
        for i in range(len(output_image)):
            sitk.WriteImage(tf[i], output_image[i])
def ROI_ProcessTranform():
    root = r"D:\USERS\yq\TH2_Part\90_110"
    fileName = os.path.join(root, "Reconstruction/ReconstructionInput.json")
    with open(fileName) as f:
        doc = json.load(f)['tasks']['process_transforms_2_1_1']
    test_temp = {i: (doc['input_targets'][i]['path']) for i in doc['input_targets']}
    output_image = [i['path'] for i in doc['output_targets']]
    param = doc['parameters']
    param['nonrigid'] = True
    # param
    print(f"start {fileName}")
    inputList = {}
    output_image = [i['path'] for i in doc['output_targets']]
    outputList = []
    old_part = 'TH2_Reconstruction\\Reconstruction\\Temp'

    # 新的路径部分
    new_part = 'TH2_Reconstruction\\ROI_76_102\\ROIReconstruction\\th2_0528'
    indexList = range(78,100)
    for key in test_temp:
        index,_,_ = key.split(',')
        index = int(index)
        if index in indexList:
            # 需要替换的旧路径部分


            # 替换路径并调整斜杠方向
            new_path = test_temp[key]
            # new_path = test_temp[key].replace(old_part, new_part)

            inputList[key] = new_path

    for i in indexList:
        new_path = output_image[2 * (i - 78)]
        new_path2 = output_image[2 * (i - 78) + 1]
        # new_path = output_image[2 * (i - 1)].replace(old_part, new_part)
        # new_path2 = output_image[2 * (i - 1) + 1].replace(old_part, new_part)
        outputList.append(new_path)

        outputList.append(new_path2)



    from VISoR_Reconstruction.reconstruction.brain_reconstruct import process_transforms_
    input_image = {i: sitk.ReadImage(inputList[i]) for i in inputList}
    tf = process_transforms_(input_image, **param)
    for i in range(len(outputList)):
        sitk.WriteImage(tf[i], outputList[i])
def create_brain():
    old_part = 'TH2_Reconstruction\\Reconstruction\\Temp'
    # 新的路径部分
    new_part = 'TH2_Reconstruction\\ROI_76_102\\ROIReconstruction\\th2_0528'
    with open(r'D:\USERS\yq\TH2_Part\90_110\Reconstruction\ReconstructionInput.json') as f:
        doc = json.load(f)['tasks']['create_brain_2_1_1']
    input_ = {}
    param = doc['parameters']

    output = doc['output_targets'][0]['path']
    # output = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\BrainTransform\visor_brain.txt"
    # todo 取 6~16 测试
    temp = doc['input_targets']
    ct = 0
    indexList = range(78, 100)
    param['internal_pixel_size'] = 1.0
    inputDict = {}
    for key in temp:
        index, _ = key.split(',')
        index = int(index)
        if index in indexList:
            # 替换路径并调整斜杠方向

            inputDict[key] = temp[key]
    for k, v in inputDict.items():
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
def _generate_brain_image():
    # br,imgPath,i,name_format,n_start = taskpara
    imgFormat = r"D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\new\test_{}.tif"
    name_format = os.path.join(r"D:\USERS\yq\TH2_Part\90_110\Reconstruction\BrainImage\1.0", 'Z{:05d}_' + 'C{}.tif'.format('1'))
    br = VISoRBrain()
    br_path = r"D:\USERS\yq\TH2_Part\90_110\Reconstruction\BrainTransform\visor_brain.txt"
    br.load(br_path)
    # print(imgPath,sliceIndex,n_start)
    for sliceIndex in range(80,100):
        imgPath = imgFormat.format(sliceIndex)
        n_start = 400 * (sliceIndex - 1)
        img = sitk.ReadImage(imgPath)
        # img = RefineImg(img,refSize,imgOrigin, lefttop)
        t_dummy = generate_brain_image(brain=br, img = img, slice_index=sliceIndex
                                       , input_pixel_size=1.0,output_pixel_size=1.0,name_format=name_format,
                                       n_start=n_start)
def RenameMha():
    # 假设这些文件位于某个特定目录中
    directory = r'D:\USERS\yq\TH2_Reconstruction\ROI_76_102\ROIReconstruction\th2_0528'

    # 遍历指定目录
    for filename in os.listdir(directory):
        if filename.endswith("_uxy.mha") or filename.endswith("_lxy.mha"):
            # 分离文件名和后缀
            name, extension = os.path.splitext(filename)

            # 提取文件名中的数字部分和类型标识（'uxy' 或 'lxy'）
            parts = name.split('_')
            index = parts[0]
            suffix = parts[1]

            # 格式化新文件名
            new_filename = f'2_1_1_{int(index):03d}_561nm_10X_{suffix}{extension}'

            # 构建完整的原始文件路径和新文件路径
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            # 重命名文件
            os.rename(old_path, new_path)
            print(f'Renamed "{filename}" to "{new_filename}"')
            print(f'Renamed "{old_path}" to "{new_path}"')

if __name__ == '__main__':
    # main()
    # RenameMha()
    # ROI_ProcessTranform()
    create_brain()
    # _generate_brain_image()

