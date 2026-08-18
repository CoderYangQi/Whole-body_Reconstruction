import gc
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
    batch_size = 1
    save_root = r"Z:\users\yq\MorphDatasets\OASIS\save_model\dust_0224"
    os.makedirs(save_root, exist_ok=True)

    train_dir = r'Z:\users\yq\MorphDatasets\OASIS\OASIS_L2R_2021_task03/All/'
    val_dir = r'Z:\users\yq\MorphDatasets\OASIS\OASIS_L2R_2021_task03/Test/'
    weights = [1, 1, 1]  # loss weights
    save_dir = 'TransMorph_ncc_{}_dsc{}_diffusion_{}/'.format(weights[0], weights[1], weights[2])
    os.makedirs(os.path.join(save_root, 'experiments/' + save_dir), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'logs/' + save_dir), exist_ok=True)
    sys.stdout = Logger(os.path.join(save_root, 'logs/' + save_dir))
    lr = 0.0001  # learning rate
    epoch_start = 0
    max_epoch = 100  # max traning epoch
    cont_training = True  # if continue training

    '''
    Initialize model
    '''

    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR

    config = CONFIGS_TM['TransMorph']

    num = 2
    model = TransMorph.RecursiveCascadeNetwork(n_cascades=num, config=config)
    model.cuda()
    trainable_params = []
    for submodel in model.stems:
        trainable_params += list(submodel.parameters())

    trainable_params += list(model.reconstruction.parameters())

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
        # epoch_start = 201
        # model_dir = 'experiments/' + save_dir
        # updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        # print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        # model.load_state_dict(best_model)

        model_dir = os.path.join(save_root, 'experiments/' + save_dir)

        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])
        # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
        print('Best model: {}'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_all_state(best_model)
        epoch_start = best_model['epoch']
        model.cuda()
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),
                                         ])

    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    train_files = glob.glob(train_dir + '*.pkl')
    val_files = glob.glob(val_dir + '*.pkl')
    train_set = datasets.OASISBrainDataset(train_files, transforms=train_composed)
    val_set = datasets.OASISBrainInferDataset(val_files, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = Adam(trainable_params, lr=2e-5)

    criterion_ncc = losses.NCC_vxm()
    # criterion_dsc = losses.DiceLoss()
    criterion_reg = losses.Grad3d(penalty='l2')
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/' + save_dir)

    num_class = 36
    del best_model
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        model.train()
        for data in train_loader:
            idx += 1
            torch.cuda.empty_cache()

            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            optimizer.zero_grad()

            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            # x_seg = data[2]
            # y_seg = data[3]



            # x_in = torch.cat((x, y), dim=1)
            # x = x.cuda()
            # y = y.cuda()
            start_model = time.time()
            warped, flows = model(x, y)
            print(f"model cal time is {time.time() - start_model}")



            # def_segs = []
            # x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=36)
            # x_seg_oh = torch.squeeze(x_seg_oh, 1)
            # x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            #
            # for i in range(36):
            #     def_seg = model.reconstruction(x_seg_oh[:, i:i + 1, ...], flows[-1])
            #     def_segs.append(def_seg.cpu())
            # def_seg = torch.cat(def_segs, dim=1)


            loss_ncc = criterion_ncc(warped[-1], y) * weights[0]

            # loss_dsc = criterion_dsc(def_seg.cpu().long(), y_seg.cpu().long(), num_class) * weights[1]
            loss_reg = sum([criterion_reg(flow, flow) for flow in flows]) * weights[2] / 2

            loss = loss_ncc + loss_reg
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            # loss.backward()
            loss.backward(retain_graph=False)  # 确保不保留计算图

            optimizer.step()

            # del x_seg_oh, x_in, def_segs, def_seg, loss
            del x,y,warped, flows
            del loss, loss_ncc, loss_reg  # 释放变量
            torch.cuda.empty_cache()
            x = data[0]
            y = data[1]

            # x = x.cuda()
            # y = y.cuda()


            warped, flows = model(y, x)


            loss_ncc = criterion_ncc(warped[-1], x) * weights[0]
            loss_reg = sum([criterion_reg(flow, flow) for flow in flows]) * weights[2] / 2
            loss = loss_ncc + loss_reg
            loss_all.update(loss.item(), x.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            del x, y, warped, flows
            gc.collect()

            loss.backward(retain_graph=False)  # 确保不保留计算图
            # loss.backward()
            optimizer.step()




            # del y_seg_oh, y_in, def_segs, def_seg
            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, DSC: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader),
                                                                                                loss.item(),
                                                                                                loss_ncc.item(),
                                                                                                0,
                                                                                                loss_reg.item()))
            del loss, loss_ncc, loss_reg  # 释放变量
            torch.cuda.empty_cache()
        # writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                warped, flows = model(x, y)
                def_out = reg_model([x_seg.cuda().float(), flows[-1].cuda()])
                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
            del x, y, warped, flows

        best_dsc = max(eval_dsc.avg, best_dsc)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_dsc': best_dsc,
        #     'optimizer': optimizer.state_dict(),
        # }, save_dir=os.path.join(save_root, 'experiments/' + save_dir),
        #     filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg))
        save_filename = 'dsc{:.4f}.pth.tar'.format(eval_dsc.avg)
        ckp = {}
        for i, submodel in enumerate(model.stems):
            ckp[f"cascade {i}"] = submodel.state_dict()

        # ckp['train_loss'] = train_loss_log
        ckp['best_dsc'] = best_dsc
        ckp['epoch'] = epoch
        ckp['optimizer'] = optimizer.state_dict()

        torch.save(ckp, os.path.join(save_root, 'experiments/' + save_dir, save_filename))
        loss_all.reset()
        del def_out, ckp
    # writer.close()


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
    GPU_iden = 3
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