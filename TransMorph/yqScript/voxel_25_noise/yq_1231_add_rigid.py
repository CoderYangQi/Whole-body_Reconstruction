import glob, sys
import json
import os
import time

import SimpleITK as sitk
import TransMorph.losses
import TransMorph.utils as utils
from torch.utils.data import DataLoader
from TransMorph.data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from TransMorph.models.TransMorph import CONFIGS as CONFIGS_TM
import TransMorph.models.TransMorph as TransMorph
from scipy.ndimage.interpolation import map_coordinates, zoom
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


def RigidTransform(fixed_path, moving_path, parameter_file, output_path):
    """
        Perform rigid transformation using SimpleITK.

        :param fixed_image_path: Path to the fixed image.
        :param moving_image_path: Path to the moving image.
        :param parameter_file: Path to the parameter file (txt format).
        :param output_path: Path to save the transformed image.
        """
    # Read fixed and moving images
    fixed_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    # Create elastix image filter
    elastix = sitk.ElastixImageFilter()

    # Set input images
    elastix.SetFixedImage(fixed_image)
    elastix.SetMovingImage(moving_image)

    # Load parameter file
    elastix.SetParameterMap(sitk.ReadParameterFile(parameter_file))

    # Perform registration
    elastix.Execute()

    # Get the transformed moving image
    result_image = elastix.GetResultImage()

    # Save the transformed image
    sitk.WriteImage(result_image, output_path)
    print(f"Transformed image saved to: {output_path}")
def name2tensor(img_path):
    img = sitk.ReadImage(img_path)
    arr = sitk.GetArrayFromImage(img)
    tensor = torch.tensor(arr.astype(np.float32), dtype=torch.float32).unsqueeze(
        0).unsqueeze(0)
    return tensor
def main():
    # test_dir = r"D:\USERS\yq\code\TransMorph\OASIS_L2R_2021_task03\Test"
    save_dir = r'Z:\users\yq\MorphDatasets\Bspine\1114Data\TestResult_Real'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    model_idx = -1
    # weights = [1, 1, 1]
    # model_folder = 'TransMorphLarge_ncc_{}_dsc_{}_diffusion_{}/'.format(weights[0], weights[1], weights[2])
    # model_dir = 'experiments/' + model_folder
    # path of models
    weights = [1, 0.02]  # loss weights
    model_folder = 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    exp_dir =r"Z:\users\yq\MorphDatasets\model\TransMorph\1114\experiments"
    model_dir = os.path.join(exp_dir, model_folder)


    # config = CONFIGS_TM['TransMorph-Large']
    # model = TransMorph.TransMorph(config)

    H, W, D = 64, 256, 256
    config = CONFIGS_TM['TransMorph']
    config.img_size = (H, W, D)
    config.window_size = (H // 16, W // 32, D // 32)
    model = TransMorph.TransMorph(config)

    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
# todo load test files
    from Loss.LossFunc import GlobalNCC,SSIMLoss,GlobalMutualInformationLoss,MSELossND
    ncc_loss = GlobalNCC()
    ssim_loss = SSIMLoss(spatial_dims=3)
    mi_loss = GlobalMutualInformationLoss()
    mse_loss = MSELossND()

    train_num = 0
    val_num = 81
    fixed_root = r"Z:\users\yq\MorphDatasets\model\TransMorph\1101\TestResult\fixed"
    moving_root = r"Z:\users\yq\MorphDatasets\model\TransMorph\1101\TestResult\moving"
    ext = '.tif'

    valid_fixed_list = get_files(fixed_root, ext)[train_num:val_num + train_num]
    valid_moving_list = get_files(moving_root, ext)[train_num:val_num + train_num]
    # todo 使用 rigid transform 来

    moved_root = r"Z:\users\yq\MorphDatasets\model\TransMorph\1101\TestResult\moved_rigid"
    os.makedirs(moved_root,exist_ok=True)
    parameter_file = r"D:\USERS\yq\code\TransMorph_Transformer\parameters\tp_registration_rigid.txt"
    # parameter_file = r"D:\USERS\yq\code\TransMorph_Transformer\parameters\tp_align_columns_2.txt"
    for fixed_path, moving_path in zip(valid_fixed_list, valid_moving_list):
        basename = os.path.basename(fixed_path)
        output_path = os.path.join(moved_root, basename)
        RigidTransform(fixed_path, moving_path, parameter_file, output_path)

        y = name2tensor(fixed_path)
        x = name2tensor(moving_path)
        out = name2tensor(output_path)
        data_range = max(y.max().unsqueeze(0), out.max().unsqueeze(0))
        # calculate global ncc
        res_ncc_loss = ncc_loss(y, out)
        res_ncc_ori_loss = ncc_loss(y, x)
        res_ssim_loss = 1 - ssim_loss(y, out, data_range)
        res_ssim_ori_loss = 1 - ssim_loss(y, x, data_range)
        print(f"res_ncc_loss is {res_ncc_loss} \nres_ncc_ori_loss is {res_ncc_ori_loss}")
        print(f"res_ssim_loss is {res_ssim_loss} \nres_ssim_ori_loss is {res_ssim_ori_loss}")

        print(f"{output_path}")






    # todo valid_moving_label_files is for taking place
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    test_set = datasets.VISoRDataset(valid_fixed_list, valid_moving_list, transforms=test_composed)
    val_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)



    loss_dict = {}
    with torch.no_grad():
        stdy_idx = 0
        for data in val_loader:
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]

            model.eval()
            x_in = torch.cat((x, y),dim=1)
            start_time = time.time()
            x_def, flow = model(x_in)
            print(f"used time is {time.time() - start_time}")
            # debug my flow test
            out = model.spatial_trans(x, flow)

            temp_dict = {}

            data_range = max(y.max().unsqueeze(0), out.max().unsqueeze(0))
            # calculate global ncc
            res_ncc_loss = ncc_loss(y, out)
            res_ncc_ori_loss = ncc_loss(y, x)
            res_ssim_loss = 1 - ssim_loss(y, out, data_range)
            res_ssim_ori_loss = 1 - ssim_loss(y, x, data_range)
            res_mi_loss = -mi_loss(y, out)
            res_mi_ori_loss = -mi_loss(y, x)
            res_mse_loss = mse_loss(y, out)
            res_mse_ori_loss = mse_loss(y, x)
            print(f"stdy_idx is {stdy_idx} / 200")
            print(f"global ncc is {res_ncc_loss}; ori ncc loss is {res_ncc_ori_loss}")
            print(f"ssim is {res_ssim_loss}; ori ssim loss is {res_ssim_ori_loss}")
            print(f"mi is {res_mi_loss}; ori mi loss is {res_mi_ori_loss}")
            print(f"mse is {res_mse_loss}; ori mse loss is {res_mse_ori_loss}")
            # save loss
            temp_dict['ncc'] = [res_ncc_loss.cpu().numpy().tolist(), res_ncc_ori_loss.cpu().numpy().tolist()]
            temp_dict['ssim'] = [res_ssim_loss.cpu().numpy().tolist(), res_ssim_ori_loss.cpu().numpy().tolist()]
            temp_dict['mi'] = [res_mi_loss.cpu().numpy().tolist(), res_mi_ori_loss.cpu().numpy().tolist()]
            temp_dict['mse'] = [res_mse_loss.cpu().numpy().tolist(), res_mse_ori_loss.cpu().numpy().tolist()]
            loss_dict[stdy_idx] = temp_dict


            test = out - x_def
            test = test.detach().cpu().numpy()

            x_def = x_def.detach().cpu().numpy()
            x_def = np.squeeze(x_def)
            x = np.squeeze(x.detach().cpu().numpy())
            y = np.squeeze(y.detach().cpu().numpy())
            out = np.squeeze(out.detach().cpu().numpy())
            flow = flow.cpu().detach().numpy()[0]
            # flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
            print(flow.shape)
            # np.savez(save_dir+'/disp_{}.npz'.format(file_name), flow)
            np.savez(os.path.join(save_dir, ('disp_{:04d}' + '.npz').format(stdy_idx)), flow)
            x = sitk.GetImageFromArray(x)
            y = sitk.GetImageFromArray(y)
            out =  sitk.GetImageFromArray(out)
            sitk.WriteImage(y, os.path.join(save_dir, ('y_{:04d}' + '.tif').format(stdy_idx)))
            sitk.WriteImage(x, os.path.join(save_dir, ('x_{:04d}' + '.tif').format(stdy_idx)))
            sitk.WriteImage(out, os.path.join(save_dir, ('out_{:04d}' + '.tif').format(stdy_idx)))
            stdy_idx += 1

    # 保存为 JSON 文件
    with open("real_trans_simple_loss.json", "w") as file:
        json.dump(loss_dict, file, indent=4)  # 使用 indent 美化格式
if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()