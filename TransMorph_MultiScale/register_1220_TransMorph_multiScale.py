import json

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
    ### todo init(
    labelNums = 16
    batch_size = 1
    train_num = 300
    val_num = 100
    labels_path = r"Z:\users\yq\MorphDatasets\Bspine\0418\labels.npy"
    fixed_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\fixed_image"
    moving_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\moving_image"
    train_fixed_list = get_files(fixed_root, '.nii')[:train_num]
    train_moving_list = get_files(moving_root, '.nii')[:train_num]

    valid_fixed_list = get_files(fixed_root, '.nii')[train_num:val_num + train_num]
    valid_moving_list = get_files(moving_root, '.nii')[train_num:val_num + train_num]

    fixed_label_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\fixed_label"
    moving_label_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\moving_label"
    valid_fixed_label_files = get_files(fixed_label_root, '.nii')[train_num:val_num + train_num]
    # fixed_label_files = sorted(fixed_label_files)
    valid_moving_label_files = get_files(moving_label_root, ".nii")[train_num:val_num + train_num]
    # train_fixed_list = fixed_img_list[:num];
    # train_moving_list = moving_img_list[:num]
    # valid_fixed_list = fixed_img_list[num:];
    # valid_moving_list = moving_img_list[num:]

    weights = [1, 0.02]  # loss weights
    save_dir = 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    exp_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\1211\experiments"
    log_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\1211\logs"
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
    model = TransMorph.RecursiveCascadeNetwork(n_cascades=num, config=config)
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
    '''
    Initialize training
    '''
    from Loss.LossFunc import GlobalNCC, SSIMLoss, GlobalMutualInformationLoss, MSELossND
    ncc_loss = GlobalNCC()
    ssim_loss = SSIMLoss(spatial_dims=3)
    mi_loss = GlobalMutualInformationLoss()
    mse_loss = MSELossND()

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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                              drop_last=True)
    loss_dict = {}
    stdy_idx = 0

    save_tif = r"Z:\users\yq\MorphDatasets\Bspine\0305_visor_pic\dust"
    os.makedirs(save_tif,exist_ok=True)
    import SimpleITK as sitk
    with torch.no_grad():
        for data in train_loader:
            temp_dict = {}
            model.eval()
            # fixed, moving = next(val_generator)
            # data = [t.cuda() for t in data]
            fixed = data[0]
            moving = data[1]
            fixed = fixed.cuda()
            moving = moving.cuda()
            warped, flows = model(moving, fixed)
            # sim, reg = total_loss(fixed, warped[-1], flows)
            sim_loss = pearson_correlation(fixed, warped[-1])
            # sim_loss = pearson_correlation(fixed, warped[-2])

            # calculate global ncc
            y = fixed
            out = warped[-1]
            x = moving
            data_range = max(y.max().unsqueeze(0), out.max().unsqueeze(0))


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

            x = np.squeeze(x.detach().cpu().numpy())
            y = np.squeeze(y.detach().cpu().numpy())
            out = np.squeeze(out.detach().cpu().numpy())
            x = sitk.GetImageFromArray(x)
            y = sitk.GetImageFromArray(y)
            out = sitk.GetImageFromArray(out)
            sitk.WriteImage(y, os.path.join(save_tif, ('y_{:04d}' + '.tif').format(stdy_idx)))
            sitk.WriteImage(x, os.path.join(save_tif, ('x_{:04d}' + '.tif').format(stdy_idx)))
            sitk.WriteImage(out, os.path.join(save_tif, ('out_{:04d}' + '.tif').format(stdy_idx)))

            stdy_idx += 1


        # 保存为 JSON 文件
    with open("double_trans_simple_loss.json", "w") as file:
        json.dump(loss_dict, file, indent=4)  # 使用 indent 美化格式



def main():
    labelNums = 16
    batch_size = 2
    train_num = 300
    val_num = 100
    labels_path = r"Z:\users\yq\MorphDatasets\Bspine\0418\labels.npy"
    fixed_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\fixed_image"
    moving_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\moving_image"
    train_fixed_list = get_files(fixed_root, '.nii')[:train_num]
    train_moving_list = get_files(moving_root, '.nii')[:train_num]

    valid_fixed_list = get_files(fixed_root, '.nii')[train_num:val_num + train_num]
    valid_moving_list = get_files(moving_root, '.nii')[train_num:val_num + train_num]

    fixed_label_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\fixed_label"
    moving_label_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\moving_label"
    valid_fixed_label_files = get_files(fixed_label_root, '.nii')[train_num:val_num + train_num]
    # fixed_label_files = sorted(fixed_label_files)
    valid_moving_label_files = get_files(moving_label_root, ".nii")[train_num:val_num + train_num]
    # train_fixed_list = fixed_img_list[:num];
    # train_moving_list = moving_img_list[:num]
    # valid_fixed_list = fixed_img_list[num:];
    # valid_moving_list = moving_img_list[num:]

    weights = [1, 0.02]  # loss weights
    save_dir = 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    exp_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\1206\experiments"
    log_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\1206\logs"
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
    # config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model.cuda()

    model_2 = TransMorph.TransMorph(config)
    model_2.cuda()
    # config = CONFIGS_TM['TransMorph']
    # model = TransMorph.TransMorph(config)
    # model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 90
        model_dir = exp_dir + '/' + save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-2])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-2]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])
    # todo
    val_composed = transforms.Compose([trans.yq_Seg_norm(),  # rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.int16)),
                                       ])
    train_set = datasets.VISoRDataset(train_fixed_list, train_moving_list, transforms=train_composed)
    val_set = datasets.VISoRSegDataset(valid_fixed_list, valid_moving_list,
                                       valid_fixed_label_files, valid_moving_label_files, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = nn.MSELoss()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/' + save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in
            del output
            # flip fixed and moving images
            loss = 0
            x_in = torch.cat((y, x), dim=1)
            output = model(x_in)
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], x) * weights[n]
                loss_vals[n] += curr_loss
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(),
                                                                                   loss_vals[0].item() / 2,
                                                                                   loss_vals[1].item() / 2))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                # x = data[0]
                # y = data[1]
                # x_seg = data[2]
                # y_seg = data[3]
                # x_in = torch.cat((x, y), dim=1)
                # grid_img = mk_grid_img(8, 1, config.img_size)
                # output = model(x_in)
                # def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                # def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                # dsc = utils.dice_val(def_out.long(), y_seg.long(), 46)
                # eval_dsc.update(dsc.item(), x.size(0))
                # print(eval_dsc.avg)

                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, 1, config.img_size)
                output = model(x_in)
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                dsc = utils.dice_val(def_out.long(), y_seg.long(), labelNums)
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir=os.path.join(exp_dir, save_dir), filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
    writer.close()


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
    GPU_iden = 5
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