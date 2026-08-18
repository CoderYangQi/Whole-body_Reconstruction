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
import SimpleITK as sitk
import json,time
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

def create_folder(name):
    if not os.path.exists(name):
        os.mkdir(name)

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    save_dir = r'Z:\users\yq\MorphDatasets\model\TransMorph\0210_semi\TestResult'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_idx = -1
    # weights = [1, 1, 1]
    # model_folder = 'TransMorphLarge_ncc_{}_dsc_{}_diffusion_{}/'.format(weights[0], weights[1], weights[2])
    # model_dir = 'experiments/' + model_folder
    # path of models
    weights = [1, 0.01, 0.01]  # loss weights
    os.makedirs(save_dir, exist_ok=True)
    exp_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\0210_semi\experiments"
    log_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\0210_semi\logs"
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

    from Loss.LossFunc import GlobalNCC,SSIMLoss,GlobalMutualInformationLoss,MSELossND
    ncc_loss = GlobalNCC()
    ssim_loss = SSIMLoss(spatial_dims=3)
    mi_loss = GlobalMutualInformationLoss()
    mse_loss = MSELossND()

    dice_loss = losses.DiceLoss(eps=1e-5)

    start_test = 0
    end_test = 200
    batch_size = 1
    test_root = r"Z:\users\yq\MorphDatasets\Bspine\0217"
    fixed_root = test_root + "\\fixed_image"
    moving_root = test_root + "\\moving_image"
    train_fixed_list = get_files(fixed_root, '.nii')[start_test:end_test]
    train_moving_list = get_files(moving_root, '.nii')[start_test:end_test]
    print(f"test data length is {len(train_fixed_list)}")
    fixed_label_root = test_root + "\\fixed_label"
    moving_label_root = test_root + "\\moving_label"
    train_fixed_label_files = get_files(fixed_label_root, '.nii')[start_test:end_test]
    train_moving_label_files = get_files(moving_label_root, ".nii")[start_test:end_test]

    train_composed = transforms.Compose([trans.yq_Seg_norm(),  # rearrange segmentation label to 1 to 46
                                         trans.NumpyType((np.float32, np.int16)),
                                         ])
    train_set = datasets.VISoRSegDataset(train_fixed_list, train_moving_list,
                                         train_fixed_label_files, train_moving_label_files,
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

            # calculate the dice loss
            x_seg = data[2]
            y_seg = data[3]
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            res_dice = 1 - dice_loss(def_out.long(), y_seg.cuda().long(), 16)
            res_ori_dice = 1 - dice_loss(x_seg.cuda().long(), y_seg.cuda().long(), 16)
            print(f"label dice is {res_dice}; ori label dice is {res_ori_dice}")


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
            temp_dict['dice'] = [res_dice.cpu().numpy().tolist(), res_ori_dice.cpu().numpy().tolist()]

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
    with open("trans_0217_semi_loss_dice.json", "w") as file:
        json.dump(loss_dict, file, indent=4)  # 使用 indent 美化格式

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

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
    main()