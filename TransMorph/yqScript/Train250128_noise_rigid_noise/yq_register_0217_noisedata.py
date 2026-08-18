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
def main():
    # test_dir = r"D:\USERS\yq\code\TransMorph\OASIS_L2R_2021_task03\Test"
    save_dir = r'Z:\users\yq\MorphDatasets\Bspine\noise_1114\TestResult'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    model_idx = -1
    # weights = [1, 1, 1]
    # model_folder = 'TransMorphLarge_ncc_{}_dsc_{}_diffusion_{}/'.format(weights[0], weights[1], weights[2])
    # model_dir = 'experiments/' + model_folder
    # path of models
    weights = [1, 0.01, 0.01]  # loss weights
    os.makedirs(save_dir,exist_ok=True)
    exp_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\noise0129\experiments"
    log_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\noise0129\logs"
    model_folder = 'TransMorph_mse_{}_diffusion_{}_dice_{}/'.format(weights[0], weights[1], weights[2])
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

    start_test = 800
    end_test = 1000
    train_num = 2000
    val_num = 20
    batch_size = 1
    # fixed_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\fixed_image"
    # moving_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\moving_image"
    fixed_root = r"Z:\users\yq\MorphDatasets\Bspine\noise_1114\fixed_image"
    moving_root = r"Z:\users\yq\MorphDatasets\Bspine\noise_1114\moving_image"
    train_fixed_list = get_files(fixed_root, '.nii')[start_test:end_test]
    train_moving_list = get_files(moving_root, '.nii')[start_test:end_test]

    train_composed = transforms.Compose([
        trans.NumpyType((np.float32, np.float32)),
    ])
    train_set = datasets.VISoRDataset(train_fixed_list, train_moving_list,
                                      transforms=train_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    loss_dict = {}
    with torch.no_grad():
        stdy_idx = 0
        for data in train_loader:
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

            stdy_idx += 1

    # 保存为 JSON 文件
    with open("trans_0217_noisedata_loss.json", "w") as file:
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