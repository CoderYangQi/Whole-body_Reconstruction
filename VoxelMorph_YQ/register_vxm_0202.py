import glob
import json

from torch.utils.tensorboard import SummaryWriter
import os, losses, utils
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models import VxmDense_1, VxmDense_2, VxmDense_huge
import SimpleITK as sitk

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
    labelNums = 16
    batch_size = 2
    train_num = 800
    val_num = 200
    batch_size = 4
    valid_batch_size = 1
    weights = [1, 0.02]
    save_dir = 'vxm_2_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    exp_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\voxelmorph\0202\experiments"
    log_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\voxelmorph\0202\logs"
    save_res = r"Z:\users\yq\MorphDatasets\model\TransMorph\voxelmorph\0202\result"
    os.makedirs(save_res,exist_ok=True)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(os.path.join(exp_dir, save_dir)):
        os.makedirs(os.path.join(exp_dir, save_dir))
    # logs
    if not os.path.exists(os.path.join(log_dir, save_dir)):
        os.makedirs(os.path.join(log_dir, save_dir))

    # if not os.path.exists('experiments/' + save_dir):
    #     os.makedirs('experiments/' + save_dir)
    # if not os.path.exists('logs/' + save_dir):
    #     os.makedirs('logs/' + save_dir)
    # sys.stdout = Logger('logs/' + save_dir)
    sys.stdout = Logger(os.path.join(log_dir,save_dir))

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
    model_folder = 'vxm_2_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_idx = -1

    model_dir = os.path.join(exp_dir, model_folder)

    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
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
    # if cont_training:
    #     epoch_start = 0
    #     # model_dir = 'experiments/'+save_dir
    #     model_dir = exp_dir + '/' + save_dir
    #     updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
    #     best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-3])['state_dict']
    #     model.load_state_dict(best_model)
    # else:
    #     updated_lr = lr
    if cont_training:
        model_dir = exp_dir + '/' + save_dir
        checkPoint = torch.load(model_dir + natsorted(os.listdir(model_dir))[-3])
        best_model = checkPoint['state_dict']
        model.load_state_dict(best_model)
        epoch_start = checkPoint['epoch']
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)

    else:
        updated_lr = lr

    '''
    Initialize training
    '''

    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(),
        trans.NumpyType((np.float32, np.int16)),
                                        ])
    start_test = 2000
    end_test = 2200
    train_num = 2000
    val_num = 20
    batch_size = 1
    # fixed_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\fixed_image"
    # moving_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\moving_image"
    fixed_root = r"Z:\users\yq\MorphDatasets\Bspine\1208\fixed_image"
    moving_root = r"Z:\users\yq\MorphDatasets\Bspine\1208\moving_image"
    train_fixed_list = get_files(fixed_root, '.nii')[start_test:end_test]
    train_moving_list = get_files(moving_root, '.nii')[start_test:end_test]


    train_composed = transforms.Compose([
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])
    train_set = datasets.VISoRDataset(train_fixed_list, train_moving_list,
                                      transforms=train_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)


    criterion = nn.MSELoss()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    # from TransMorph.yqScript.Train1114.GlobalNCC import GlobalNCC
    from Loss.LossFunc import GlobalNCC,SSIMLoss,GlobalMutualInformationLoss,MSELossND
    ncc_loss = GlobalNCC()
    ssim_loss = SSIMLoss(spatial_dims=3)
    mi_loss = GlobalMutualInformationLoss()
    mse_loss = MSELossND()

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
            # x_seg = data[2]
            # y_seg = data[3]
            x_in = torch.cat((x, y), dim=1)
            x_def, flow = model(x_in)

            # output = model(x_in)
            # out = reg_model([x.cuda().float(), flow.cuda()])
            out = model.transformer(x.cuda().float(), flow.cuda())
            temp_dict = {}

            data_range = max(y.max().unsqueeze(0), out.max().unsqueeze(0))
            # calculate global ncc
            res_ncc_loss = ncc_loss(y, out)
            res_ncc_ori_loss = ncc_loss(y, x)
            res_ssim_loss = 1 - ssim_loss(y, out,data_range)
            res_ssim_ori_loss = 1 - ssim_loss(y, x,data_range)
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
            temp_dict['ncc'] = [res_ncc_loss.cpu().numpy().tolist(),res_ncc_ori_loss.cpu().numpy().tolist()]
            temp_dict['ssim'] = [res_ssim_loss.cpu().numpy().tolist(),res_ssim_ori_loss.cpu().numpy().tolist()]
            temp_dict['mi'] = [res_mi_loss.cpu().numpy().tolist(),res_mi_ori_loss.cpu().numpy().tolist()]
            temp_dict['mse'] = [res_mse_loss.cpu().numpy().tolist(),res_mse_ori_loss.cpu().numpy().tolist()]
            test = out - x_def
            test = test.detach().cpu().numpy()

            loss_dict[stdy_idx] = temp_dict

            x_def = x_def.detach().cpu().numpy()
            x_def = np.squeeze(x_def)
            x = np.squeeze(x.detach().cpu().numpy())
            y = np.squeeze(y.detach().cpu().numpy())
            out = np.squeeze(out.detach().cpu().numpy())
            flow = flow.cpu().detach().numpy()[0]
            # flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
            print(flow.shape)
            # np.savez(save_dir+'/disp_{}.npz'.format(file_name), flow)
            np.savez(os.path.join(save_res, ('disp_{:04d}' + '.npz').format(stdy_idx)), flow)
            x = sitk.GetImageFromArray(x)
            y = sitk.GetImageFromArray(y)
            out = sitk.GetImageFromArray(out)
            sitk.WriteImage(y, os.path.join(save_res, ('y_{:04d}' + '.tif').format(stdy_idx)))
            sitk.WriteImage(out, os.path.join(save_res, ('out_{:04d}' + '.tif').format(stdy_idx)))
            stdy_idx += 1

    # 保存为 JSON 文件
    with open("vxm_noise_0202_loss.json", "w") as file:
        json.dump(loss_dict, file, indent=4)  # 使用 indent 美化格式

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