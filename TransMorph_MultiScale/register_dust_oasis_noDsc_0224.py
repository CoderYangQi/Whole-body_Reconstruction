import gc
import json
import time

from torch.utils.tensorboard import SummaryWriter
import os, glob
import TransMorph.losses as losses
import TransMorph.utils as utils
import sys
from torch.utils.data import DataLoader
from TransMorph.data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from TransMorph.models.TransMorph import CONFIGS as CONFIGS_TM
import TransMorph.models.TransMorph as TransMorph
from TransMorph_MultiScale.losses import total_loss,pearson_correlation,regularize_loss_3d


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
    return image_files_path


def create_folder(name):
    if not os.path.exists(name):
        os.mkdir(name)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def TestNulti():
    # test_dir = r'Z:\users\yq\MorphDatasets\OASIS\OASIS_L2R_2021_task03\Challenge_test_no_gt\\'
    test_dir = r'Z:\users\yq\MorphDatasets\OASIS\OASIS_L2R_2021_task03\test\\'
    save_dir = r'Z:\users\yq\MorphDatasets\OASIS\OASIS_L2R_2021_task03\dust_0224_noDsc'
    # train_root = r"Z:\users\yq\MorphDatasets\OASIS\save_model\vxm_0224\experiments"
    os.makedirs(save_dir, exist_ok=True)
    model_idx = -1
    weights = [1, 1, 1]
    model_folder = 'TransMorph_ncc_{}_dsc_{}_diffusion_{}/'.format(weights[0], weights[1], weights[2])
    # model_dir = os.path.join(train_root, model_folder)
    model_dir = r"Z:\users\yq\MorphDatasets\OASIS\save_model\dust_0224\experiments\TransMorph_ncc_1_dsc1_diffusion_1\\"
    config = CONFIGS_TM['TransMorph']

    num = 2
    model = TransMorph.RecursiveCascadeNetwork(n_cascades=num, config=config)
    # model = TransMorph.TransMorph(config)
    # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])
    # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']

    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_all_state(best_model)
    model.cuda()
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)), ])
    test_set = datasets.OASISBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    file_names = glob.glob(test_dir + '*.pkl')

    '''
    init all loss func
    '''
    from Loss.LossFunc import GlobalNCC, SSIMLoss, GlobalMutualInformationLoss, MSELossND, DiceLoss
    ncc_loss = GlobalNCC()
    ssim_loss = SSIMLoss(spatial_dims=3)
    mi_loss = GlobalMutualInformationLoss()
    mse_loss = MSELossND()
    dice_loss = DiceLoss(eps=1e-5)

    loss_dict = {}
    with torch.no_grad():
        stdy_idx = 0
        for data in file_names:
            temp_dict = {}
            x, y, x_seg, y_seg = utils.pkload(data)
            x_seg, y_seg = x_seg[None, None, ...], y_seg[None, None, ...]
            x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
            y_seg = np.ascontiguousarray(y_seg)
            x_seg, y_seg = torch.from_numpy(x_seg), torch.from_numpy(y_seg)

            # x, y = utils.pkload(data)
            x, y = x[None, None, ...], y[None, None, ...]
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
            file_name = file_names[stdy_idx].split('\\')[-1].split('.')[0][2:]
            print(file_name)
            model.eval()
            start_time = time.time()

            warped, flows = model(x, y)

            used_time = time.time() - start_time
            print(f"used time is {used_time}")

            # todo calculate dsc
            def_out = reg_model([x_seg.cuda().float(), flows[-1].cuda()])
            res_dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
            res_dsc_ori = utils.dice_val_VOI(x_seg.long(), y_seg.long())

            out = warped[-1]
            data_range = max(y.max().unsqueeze(0), out.max().unsqueeze(0))

            res_ncc_loss = ncc_loss(y, out)
            res_ncc_ori_loss = ncc_loss(y, x)
            res_ssim_loss = 1 - ssim_loss(y, out, data_range)
            res_ssim_ori_loss = 1 - ssim_loss(y, x, data_range)
            res_mi_loss = -mi_loss(y, out)
            res_mi_ori_loss = -mi_loss(y, x)
            res_mse_loss = mse_loss(y, out)
            res_mse_ori_loss = mse_loss(y, x)

            res_dice = 1 - dice_loss(def_out.long(), y_seg.cuda().long(), 36)
            res_ori_dice = 1 - dice_loss(x_seg.cuda().long(), y_seg.cuda().long(), 36)
            print(f"label dice is {res_dice}; ori label dice is {res_ori_dice}")

            # eval_dsc.update(dsc.item(), x.size(0))
            print(f"dsc is {res_dsc}; ori dsc loss is {res_dsc_ori}")

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
            temp_dict['dice'] = [res_dice.cpu().numpy().tolist(), res_ori_dice.cpu().numpy().tolist()]
            loss_dict[stdy_idx] = temp_dict

            # np.savez(save_dir + 'disp_{}.npz'.format(file_name), flow)
            stdy_idx += 1

            # 保存为 JSON 文件
        with open("dust_oasis_noDsc_0327_loss.json", "w") as file:
            json.dump(loss_dict, file, indent=4)  # 使用 indent 美化格式

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 6
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    # main()
    TestNulti()