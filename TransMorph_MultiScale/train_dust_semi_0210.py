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
    train_num = 800
    val_num = 200
    labels_path = r"Z:\users\yq\MorphDatasets\Bspine\0418\labels.npy"
    fixed_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\fixed_image"
    moving_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\moving_image"
    train_fixed_list = get_files(fixed_root, '.nii')[:train_num]
    train_moving_list = get_files(moving_root, '.nii')[:train_num]

    fixed_label_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\fixed_label"
    moving_label_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\moving_label"
    train_fixed_label_files = get_files(fixed_label_root, '.nii')[:train_num]
    train_moving_label_files = get_files(moving_label_root, ".nii")[:train_num]


    valid_fixed_list = get_files(fixed_root, '.nii')[train_num:val_num + train_num]
    valid_moving_list = get_files(moving_root, '.nii')[train_num:val_num + train_num]



    valid_fixed_label_files = get_files(fixed_label_root, '.nii')[train_num:val_num + train_num]
    # fixed_label_files = sorted(fixed_label_files)
    valid_moving_label_files = get_files(moving_label_root, ".nii")[train_num:val_num + train_num]
    # train_fixed_list = fixed_img_list[:num];
    # train_moving_list = moving_img_list[:num]
    # valid_fixed_list = fixed_img_list[num:];
    # valid_moving_list = moving_img_list[num:]

    weights = [1, 0.01, 0.01]  # loss weights
    save_dir = 'TransMorph_mse_{}_diffusion_{}_dice_{}/'.format(weights[0], weights[1], weights[2])
    exp_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\0210_dust_semi\experiments"
    log_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\0210_dust_semi\logs"

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

    mse_loss = nn.MSELoss()
    grad_loss = losses.Grad3d(penalty='l2')
    dice_loss = losses.DiceLoss(eps=1e-5)


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
    '''
        Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

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

    train_set = datasets.VISoRSegDataset(train_fixed_list, train_moving_list,
                                         train_fixed_label_files, train_moving_label_files, transforms=train_composed)

    val_set = datasets.VISoRSegDataset(valid_fixed_list, valid_moving_list,
                                       valid_fixed_label_files, valid_moving_label_files, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    ### end

    optim = Adam(trainable_params, lr=1e-4)
    scheduler = StepLR(optimizer=optim, step_size=20, gamma=0.96)
    # train_generator = iter(sample_generator('./train.txt', batch_size=args.b))
    # val_generator = iter(sample_generator('./validation.txt', batch_size=args.b))

    # Saving the losses
    train_loss_log = []
    reg_loss_log = []
    val_loss_log = []
    best_loss = 1000
    for epoch in range(1, max_epoch):
        print(f"-----Epoch {epoch} / {max_epoch}-----")
        train_epoch_loss = 0
        train_reg_loss = 0
        vis_batch = []
        loss_all = utils.AverageMeter()
        model.train()
        idx = 0
        for data in train_loader:
            idx += 1
            # for iteration in range(1, args.i):
            #     if iteration % int(0.1 * args.i) == 0:
            #         print(f"\t-----Iteration {iteration} / {args.i} -----")
            optim.zero_grad()
            # fixed, moving = next(train_generator)
            # x is fixed, moving is y
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            loss_val = []

            x = x.cuda()
            y = y.cuda()
            warped, flows = model(x, y)
            # loss = total_loss(fixed, warped, flows)

            # sim_loss = pearson_correlation(fixed, warped[-1])
            loss1 = mse_loss(y, warped[-1])
            loss2 = sum([grad_loss(flow, flow) for flow in flows]) # grad_loss 第二個參數沒有意義
            # add semi supervise
            def_out = reg_model([x_seg.cuda().float(), flows[-1].cuda()])
            # def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
            loss3 = dice_loss(def_out.long(), y_seg.cuda().long(), labelNums)
            print(f"dice_loss is {loss3}")

            loss_val.append(loss1 * weights[0])
            loss_val.append(loss2 * weights[1])
            loss_val.append(loss3 * weights[2])

            # sim_loss = pearson_correlation(fixed, warped[-2])
            # reg_loss = sum([regularize_loss_3d(flow) for flow in flows])
            #
            loss = loss1 * weights[0] + loss2 * weights[1] + loss3 * weights[2]
            loss.backward()
            optim.step()
            loss_all.update(loss.item(), y.numel())

            ## todo 将数据x 和 y对调重新再算一边
            del x,y,warped,flows,def_out
            x = data[0]
            y = data[1]
            x = x.cuda()
            y = y.cuda()
            warped, flows = model(y, x)

            loss1 = mse_loss(x, warped[-1])
            loss2 = sum([grad_loss(flow, flow) for flow in flows])  # grad_loss 第二個參數沒有意義
            # add semi supervise
            def_out = reg_model([y_seg.cuda().float(), flows[-1].cuda()])
            loss3 = dice_loss(def_out.long(), x_seg.cuda().long(), labelNums)
            print(f"dice_loss is {loss3}")
            loss = loss1 * weights[0] + loss2 * weights[1] + loss3 * weights[2]
            loss.backward()
            optim.step()
            loss_all.update(loss.item(), y.numel())

            loss_val[0] += loss1 * weights[0]
            loss_val[1] += loss2 * weights[1]
            loss_val[2] += loss3 * weights[2]

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, Dice: {:.6f}'.format(idx, len(train_loader), loss.item(),
                                                                                   loss_val[0].item()/2,
                                                                                   loss_val[1].item()/2,
                                                                                   loss_val[2].item()/2))

            # if iteration == args.fixed_sample:
            #     vis_batch.append(fixed)
            #     vis_batch.append(moving)
            #     vis_batch.append(warped)
            #     vis_batch.append(flows)

        train_loss_log.append(train_epoch_loss)
        reg_loss_log.append(train_reg_loss)

        model.eval()
        print(f">>>>> Validation <<<<<")

        val_epoch_loss = 0
        val_idx = 0
        for data in val_loader:
            val_idx += 1
            # for iteration in range(1, args.iv):
        #     if iteration % int(0.1 * args.iv) == 0:
        #         print(f"\t-----Iteration {iteration} / {args.iv} -----")

            with torch.no_grad():
                # fixed, moving = next(val_generator)
                # data = [t.cuda() for t in data]
                fixed = data[0]
                moving = data[1]
                fixed = fixed.cuda()
                moving = moving.cuda()
                warped, flows = model(fixed, moving)
                # sim, reg = total_loss(fixed, warped[-1], flows)
                sim_loss = pearson_correlation(fixed, warped[-1])
                # sim_loss = pearson_correlation(fixed, warped[-2])
                reg_loss = sum([regularize_loss_3d(flow) for flow in flows])
                loss = sim_loss + reg_loss
                val_epoch_loss += loss.item()
                print('Validation: Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader),
                                                                                       loss.item(),
                                                                                       sim_loss.item(),
                                                                                       reg_loss.item()))

        val_loss_log.append(val_epoch_loss )

        scheduler.step()
        # best_dsc = max(eval_dsc.avg, best_dsc)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_dsc': best_dsc,
        #     'optimizer': optimizer.state_dict(),
        # }, save_dir=os.path.join(exp_dir, save_dir), filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))

        best_loss = min(val_epoch_loss, best_loss)
        if best_loss <= val_epoch_loss:
            ckp = {}
            for i, submodel in enumerate(model.stems):
                ckp[f"cascade {i}"] = submodel.state_dict()

            ckp['train_loss'] = train_loss_log
            ckp['val_loss'] = val_loss_log
            ckp['epoch'] = epoch

            torch.save(ckp, os.path.join(exp_dir,save_dir,'{:.3f}_epoch_{}.pth'.format(best_loss,epoch)))

        # generate_plots(vis_batch[0], vis_batch[1], vis_batch[2], vis_batch[3], train_loss_log, val_loss_log,
        #                reg_loss_log, epoch)


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