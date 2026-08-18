import glob, sys
import os
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
    from GlobalNCC import GlobalNCC
    ncc_loss = GlobalNCC()
    # test_dir = r"D:\USERS\yq\code\TransMorph\OASIS_L2R_2021_task03\Test"
    save_dir = r'Z:\users\yq\MorphDatasets\Bspine\1114Data\TestResult_1208'
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
    start_test = 2000
    end_test = 2100
    train_num = 2000
    val_num = 20
    batch_size = 1
    # fixed_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\fixed_image"
    # moving_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\moving_image"
    fixed_root = r"Z:\users\yq\MorphDatasets\Bspine\1208\fixed_image"
    moving_root = r"Z:\users\yq\MorphDatasets\Bspine\1208\moving_image"
    train_fixed_list = get_files(fixed_root, '.nii')[start_test:end_test]
    train_moving_list = get_files(moving_root, '.nii')[start_test:end_test]

    valid_fixed_list = get_files(fixed_root, '.nii')[train_num:val_num + train_num]
    valid_moving_list = get_files(moving_root, '.nii')[train_num:val_num + train_num]

    fixed_label_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\fixed_label"
    moving_label_root = r"Z:\users\yq\MorphDatasets\Bspine\1114Data\moving_label"
    valid_fixed_label_files = get_files(fixed_label_root, '.nii')[train_num:val_num + train_num]
    # fixed_label_files = sorted(fixed_label_files)
    valid_moving_label_files = get_files(moving_label_root, ".nii")[train_num:val_num + train_num]

    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),])
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])
    train_set = datasets.VISoRDataset(train_fixed_list, train_moving_list,
                                         transforms=train_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    # test_set = datasets.VISoRSegDataset(valid_fixed_list, valid_moving_list,
    #                                    valid_fixed_label_files, valid_moving_label_files, transforms=test_composed)
    # val_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    with torch.no_grad():
        stdy_idx = 0
        for data in train_loader:
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]

            model.eval()
            # x_in = torch.cat((x, y),dim=1)
            # x_def, flow = model(x_in)
            #
            # # calculate global ncc
            # loss = ncc_loss(y, x_def)
            # ori_loss = ncc_loss(y, x)
            # print(f"fisrt global ncc is {loss}; ori_loss is {ori_loss}")
            #
            # x = x_def


            # second try
            x_in = torch.cat((x, y),dim=1)
            x_def, flow = model(x_in)

            # debug my flow test out == x_def
            out = model.spatial_trans(x, flow)

            # calculate global ncc
            loss = ncc_loss(y, x_def)
            ori_loss = ncc_loss(y, x)
            print(f"second global ncc is {loss}; ori_loss is {ori_loss}")
            if loss < 0.95:
                print(f"bad case is {loss}; ori_loss is {ori_loss}")
                # import pdb; pdb.set_trace()

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
            sitk.WriteImage(out, os.path.join(save_dir, ('out_{:04d}' + '.tif').format(stdy_idx)))
            stdy_idx += 1


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