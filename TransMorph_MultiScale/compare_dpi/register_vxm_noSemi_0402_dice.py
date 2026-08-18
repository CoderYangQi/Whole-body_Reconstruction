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
import sys
from torch.utils.data import DataLoader
# from data import datasets, trans
from TransMorph.data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from VoxelMorph_YQ.models import VxmDense_1, VxmDense_2, VxmDense_huge
import SimpleITK as sitk
import json,time
class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
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
    weights = [1, 0.01, 0.01]  # loss weights
    exp_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\voxelmorph\0211_no_semi\experiments"
    log_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\voxelmorph\0211_no_semi\logs"
    save_res = r"Z:\users\yq\MorphDatasets\model\TransMorph\voxelmorph\0211_no_semi\results"

    save_dir = r'Z:\users\yq\MorphDatasets\Bspine\0402\vxmResult'
    os.makedirs(save_dir, exist_ok=True)

    os.makedirs(save_res, exist_ok=True)



    lr = 0.0001
    epoch_start = 0
    max_epoch = 200
    img_size = (64, 256, 256)
    cont_training = True

    '''
    Initialize model
    '''
    model = VxmDense_2(img_size)

    # load model
    weights = [1, 0.02, 0]
    model_folder = 'TransMorph_mse_{}_diffusion_{}_dice_{}/'.format(weights[0], weights[1], weights[2])
    # model_folder = 'vxm_2_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_idx = -1

    model_dir = os.path.join(exp_dir, model_folder)

    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print(f"load path is : {model_dir + natsorted(os.listdir(model_dir))[model_idx]}")
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)

    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(img_size, 'bilinear')
    reg_model_bilin.cuda()
    '''
            If continue from previous training
            '''


    from Loss.LossFunc import GlobalNCC, SSIMLoss, GlobalMutualInformationLoss, MSELossND,DiceLoss
    ncc_loss = GlobalNCC()
    ssim_loss = SSIMLoss(spatial_dims=3)
    mi_loss = GlobalMutualInformationLoss()
    mse_loss = MSELossND()
    dice_loss = DiceLoss(eps=1e-5)

    source_root = r"Z:\users\yq\MorphDatasets\Bspine\1231"
    fixed_root = os.path.join(source_root, "fixed_image")
    fixed_label_root = os.path.join(source_root, "fixed_label")
    moving_root = os.path.join(source_root, "moving_image")
    moving_label_root = os.path.join(source_root, "moving_label")

    start = 400
    end = 500
    test_fixed_list = get_files(fixed_root, '.nii')[start:end]
    test_moving_list = get_files(moving_root, '.nii')[start:end]

    test_fixed_label_files = get_files(fixed_label_root, '.nii')[start:end]
    test_moving_label_files = get_files(moving_label_root, ".nii")[start:end]

    batch_size = 1
    train_composed = transforms.Compose([trans.yq_Seg_norm(),  # rearrange segmentation label to 1 to 46
                                         trans.NumpyType((np.float32, np.int16)),
                                         ])
    train_set = datasets.VISoRSegDataset(test_fixed_list, test_moving_list,
                                         test_fixed_label_files, test_moving_label_files,
                                         transforms=train_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                              drop_last=True)

    '''
    Validation
    '''
    eval_dsc = utils.AverageMeter()
    loss_dict = {}
    stdy_idx = 0
    with torch.no_grad():
        for data in train_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]

            x_seg = data[2]
            y_seg = data[3]

            # 获取图像的尺寸
            batch_size, c, d, h, w = x.shape
            start = time.time()
            # 拆分 x 和 y 为四个区域（这里依赖于深度d、宽度w和高度h）
            x_tl = x[:, :, 0:d, 0:h // 2, 0:w // 2]  # 左上角
            x_tr = x[:, :, 0:d, 0:h // 2, w // 2:]  # 右上角
            x_bl = x[:, :, 0:d, h // 2:, 0:w // 2]  # 左下角
            x_br = x[:, :, 0:d, h // 2:, w // 2:]  # 右下角

            y_tl = y[:, :, 0:d, 0:h // 2, 0:w // 2]  # 左上角
            y_tr = y[:, :, 0:d, 0:h // 2, w // 2:]  # 右上角
            y_bl = y[:, :, 0:d, h // 2:, 0:w // 2]  # 左下角
            y_br = y[:, :, 0:d, h // 2:, w // 2:]

            x_seg_tl = x_seg[:, :, 0:d, 0:h // 2, 0:w // 2]  # 左上角
            x_seg_tr = x_seg[:, :, 0:d, 0:h // 2, w // 2:]  # 右上角
            x_seg_bl = x_seg[:, :, 0:d, h // 2:, 0:w // 2]  # 左下角
            x_seg_br = x_seg[:, :, 0:d, h // 2:, w // 2:]  # 右下角

            # x_in 需要是一个通道数是 2*原来通道数（因为 x 和 y 的部分拼接）
            x_in_tl = torch.cat((x_tl, y_tl), dim=1)  # 左上角拼接
            x_in_tr = torch.cat((x_tr, y_tr), dim=1)  # 右上角拼接
            x_in_bl = torch.cat((x_bl, y_bl), dim=1)  # 左下角拼接
            x_in_br = torch.cat((x_br, y_br), dim=1)  # 右下角拼接

            model.eval()
            x_in = torch.cat((x, y), dim=1)
            start_time = time.time()

            x_def_tl, flow_tl = model(x_in_tl)  # 预测左上角
            def_out_tl = reg_model([x_seg_tl.cuda().float(), flow_tl.cuda()])

            x_def_tr, flow_tr = model(x_in_tr)  # 预测右上角
            def_out_tr = reg_model([x_seg_tr.cuda().float(), flow_tr.cuda()])

            x_def_bl, flow_bl = model(x_in_bl)  # 预测左下角
            def_out_bl = reg_model([x_seg_bl.cuda().float(), flow_bl.cuda()])

            x_def_br, flow_br = model(x_in_br)  # 预测右下角
            def_out_br = reg_model([x_seg_br.cuda().float(), flow_br.cuda()])

            out_tl = model.transformer(x_tl, flow_tl)
            out_tr = model.transformer(x_tr, flow_tr)
            out_bl = model.transformer(x_bl, flow_bl)
            out_br = model.transformer(x_br, flow_br)

            # 合并四个输出（按宽度拼接左右部分，再按高度拼接上下部分）
            top_half = torch.cat((out_tl, out_tr), dim=4)  # 按宽度拼接
            bottom_half = torch.cat((out_bl, out_br), dim=4)  # 按宽度拼接

            # 继续拼接上下部分
            out = torch.cat((top_half, bottom_half), dim=3)  # 按高度拼接
            print(f"used time is {time.time() -start}")

            # 合并四个输出（按宽度拼接左右部分，再按高度拼接上下部分）
            def_top_half = torch.cat((def_out_tl, def_out_tr), dim=4)  # 按宽度拼接
            def_bottom_half = torch.cat((def_out_bl, def_out_br), dim=4)  # 按宽度拼接

            # 继续拼接上下部分
            def_out = torch.cat((def_top_half, def_bottom_half), dim=3)  # 按高度拼接

            res_dice = 1 - dice_loss(def_out.long(), y_seg.cuda().long(), 16)
            res_ori_dice = 1 - dice_loss(x_seg.cuda().long(), y_seg.cuda().long(), 16)
            print(f"label dice is {res_dice}; ori label dice is {res_ori_dice}")

            print(f"used time is {time.time() - start_time}")

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
            temp_dict['dice'] = [res_dice.cpu().detach().numpy().tolist(), res_ori_dice.cpu().detach().numpy().tolist()]

            loss_dict[stdy_idx] = temp_dict

            x = np.squeeze(x.detach().cpu().numpy())
            y = np.squeeze(y.detach().cpu().numpy())
            out = np.squeeze(out.detach().cpu().numpy())
            x = sitk.GetImageFromArray(x)
            y = sitk.GetImageFromArray(y)
            out = sitk.GetImageFromArray(out)
            sitk.WriteImage(y, os.path.join(save_dir, ('y_{:04d}' + '.tif').format(stdy_idx)))
            sitk.WriteImage(x, os.path.join(save_dir, ('x_{:04d}' + '.tif').format(stdy_idx)))
            sitk.WriteImage(out, os.path.join(save_dir, ('out_{:04d}' + '.tif').format(stdy_idx)))
            stdy_idx += 1

            # 保存为 JSON 文件
        # 使用普通transmorph 独立检测一个大图中的各个部分，然后将结果合并在一起
    with open("vxm_dpi_depart_loss_0402.json", "w") as file:
        json.dump(loss_dict, file, indent=4)  # 使用 indent 美化格式

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 2
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