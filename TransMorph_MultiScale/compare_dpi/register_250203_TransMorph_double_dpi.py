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

import SimpleITK as sitk
def TestNulti():
    save_flag = True
    ### todo init(
    labelNums = 16
    batch_size = 1
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



    weights = [1, 0.02]  # loss weights
    save_dir = 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    exp_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\0103\experiments"
    log_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\0103\logs"
    res_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\0403\tests"
    os.makedirs(res_dir,exist_ok=True)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(os.path.join(exp_dir, save_dir)):
        os.makedirs(os.path.join(exp_dir, save_dir))
    # logs
    if not os.path.exists(os.path.join(log_dir, save_dir)):
        os.makedirs(os.path.join(log_dir, save_dir))
    sys.stdout = Logger(os.path.join(log_dir, save_dir))
    lr = 0.0001  # learning rate
    epoch_start = 0
    max_epoch = 200  # max traning epoch
    args_c = 10
    cont_training = False  # if continue training

    '''
    Initialize model
    '''
    # todo change HWD
    # H, W, D = 160, 192, 224
    H, W, D = 64, 256, 256

    config = CONFIGS_TM['TransMorph']
    config.img_size = (H, W, D)
    config.window_size = (H // 16, W // 32, D // 32)
    config.use_checkpoint = False

    ### todo end  init()

    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR

    num = 2
    model = TransMorph.RecursiveCascadeNetwork_512(n_cascades=num, config=config)
    trainable_params = []
    for submodel in model.stems:
        trainable_params += list(submodel.parameters())

    trainable_params += list(model.reconstruction.parameters())

    # load model parameters
    model_idx = -1
    model_folder = save_dir

    model_dir = os.path.join(exp_dir, model_folder)

    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])
    # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    # model.load_state_dict(best_model)
    # overwrite the load function
    model.load_all_state(best_model)
    model.cuda()

    reg_model = utils.register_model((64, 512, 512), 'nearest')
    reg_model.cuda()
    '''
    Initialize training
    '''
    ## loss function

    from Loss.LossFunc import GlobalNCC, SSIMLoss, GlobalMutualInformationLoss, MSELossND,DiceLoss
    ncc_loss = GlobalNCC()
    ssim_loss = SSIMLoss(spatial_dims=3)
    mi_loss = GlobalMutualInformationLoss()
    mse_loss = MSELossND()
    dice_loss = DiceLoss(eps=1e-5)


    ### end


    # Saving the losses
    train_loss_log = []
    reg_loss_log = []
    val_loss_log = []
    best_loss = 1000
    loss_dict = {}
    idx = 0
    stdy_idx = 0
    with torch.no_grad():
        for data in train_loader:
            model.eval()
            temp_dict = {}

            x = data[0]
            y = data[1]

            x_seg = data[2]
            y_seg = data[3]
            y = y.cuda()
            x = x.cuda()
            start = time.time()
            warped, flows = model(x, y)
            # sim, reg = total_loss(fixed, warped[-1], flows)
            # sim_loss = pearson_correlation(y, warped[-1])
            # sim_loss = pearson_correlation(fixed, warped[-2])

            # calculate global ncc
            # y = fixed
            out = warped[-1]
            print(f"used time is {time.time() -start}")


            def_out = reg_model([x_seg.cuda().float(), flows[-2].cuda()])
            res_dice = 1 - dice_loss(def_out.long(), y_seg.cuda().long(), 16)
            # diff = torch.sum(def_out - x_seg)
            res_ori_dice = 1 - dice_loss(x_seg.cuda().long(), y_seg.cuda().long(), 16)
            print(f"label dice is {res_dice}; ori label dice is {res_ori_dice}")

            # if save_flag:
            #     x_ = np.squeeze(x.detach().cpu().numpy())
            #     y_ = np.squeeze(y.detach().cpu().numpy())
            #     out_ = np.squeeze(out.detach().cpu().numpy())
            #     x_ = sitk.GetImageFromArray(x_)
            #     y_ = sitk.GetImageFromArray(y_)
            #     out_ = sitk.GetImageFromArray(out_)
            #     sitk.WriteImage(y_, os.path.join(res_dir, ('y_{:04d}' + '.tif').format(stdy_idx)))
            #     sitk.WriteImage(x_, os.path.join(res_dir, ('x_{:04d}' + '.tif').format(stdy_idx)))
            #     sitk.WriteImage(out_, os.path.join(res_dir, ('out_{:04d}' + '.tif').format(stdy_idx)))
            #     del x_, y_, out_
            #     gc.collect()
            # x = moving
            data_range = max(y.max().unsqueeze(0), out.max().unsqueeze(0))
            stdy_idx += 1

            res_ncc_loss = ncc_loss(y, out)
            res_ncc_ori_loss = ncc_loss(y, x)
            res_ssim_loss = 1 - ssim_loss(y, out, data_range)
            res_ssim_ori_loss = 1 - ssim_loss(y, x, data_range)
            res_mi_loss = -mi_loss(y, out)
            res_mi_ori_loss = -mi_loss(y, x)
            res_mse_loss = mse_loss(y, out)
            res_mse_ori_loss = mse_loss(y, x)
            del x,y,out,warped,flows
            torch.cuda.empty_cache()
            print(f"stdy_idx is {stdy_idx} / 200")
            print(f"global ncc is {res_ncc_loss}; ori ncc loss is {res_ncc_ori_loss}")
            print(f"ssim is {res_ssim_loss}; ori ssim loss is {res_ssim_ori_loss}")
            print(f"mi is {res_mi_loss}; ori mi loss is {res_mi_ori_loss}")
            print(f"mse is {res_mse_loss}; ori mse loss is {res_mse_ori_loss}")
            # save loss
            temp_dict['ncc'] = [res_ncc_loss.cpu().detach().numpy().tolist(), res_ncc_ori_loss.cpu().detach().numpy().tolist()]
            temp_dict['ssim'] = [res_ssim_loss.cpu().detach().numpy().tolist(), res_ssim_ori_loss.cpu().detach().numpy().tolist()]
            temp_dict['mi'] = [res_mi_loss.cpu().detach().numpy().tolist(), res_mi_ori_loss.cpu().detach().numpy().tolist()]
            temp_dict['mse'] = [res_mse_loss.cpu().detach().numpy().tolist(), res_mse_ori_loss.cpu().detach().numpy().tolist()]
            temp_dict['dice'] = [res_dice.cpu().detach().numpy().tolist(), res_ori_dice.cpu().detach().numpy().tolist()]
            loss_dict[stdy_idx] = temp_dict

    with open("double_dpi_dust_0402.json", "w") as file:
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
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 7
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)

    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    # main()
    TestNulti()