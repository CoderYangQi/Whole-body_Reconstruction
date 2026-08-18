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
    # test_dir = r"D:\USERS\yq\code\TransMorph\OASIS_L2R_2021_task03\Test"
    save_dir = r'Z:\users\yq\MorphDatasets\model\TransMorph\0426\test_result'

    model_idx = -1
    # weights = [1, 1, 1]
    # model_folder = 'TransMorphLarge_ncc_{}_dsc_{}_diffusion_{}/'.format(weights[0], weights[1], weights[2])
    # model_dir = 'experiments/' + model_folder
    # path of models
    weights = [1, 0.02]  # loss weights
    model_folder = 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    exp_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\0426\experiments"
    log_dir = r"Z:\users\yq\MorphDatasets\model\TransMorph\0426\logs"
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

    train_num = 200
    val_num = 5
    fixed_root = r"Z:\users\yq\MorphDatasets\Bspine\0418\fixed_image"
    moving_root = r"Z:\users\yq\MorphDatasets\Bspine\0418\moving_image"
    train_fixed_list = get_files(fixed_root, '.nii')[:train_num]
    train_moving_list = get_files(moving_root, '.nii')[:train_num]

    valid_fixed_list = get_files(fixed_root, '.nii')[train_num:val_num + train_num]
    valid_moving_list = get_files(moving_root, '.nii')[train_num:val_num + train_num]

    fixed_label_root = r"Z:\users\yq\MorphDatasets\Bspine\0418\fixed_label"
    moving_label_root = r"Z:\users\yq\MorphDatasets\Bspine\0418\moving_label"
    valid_fixed_label_files = get_files(fixed_label_root, '.nii')[train_num:val_num + train_num]
    # fixed_label_files = sorted(fixed_label_files)
    valid_moving_label_files = get_files(moving_label_root, ".nii")[train_num:val_num + train_num]
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),])
    test_set = datasets.VISoRSegDataset(valid_fixed_list, valid_moving_list,
                                       valid_fixed_label_files, valid_moving_label_files, transforms=test_composed)
    val_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),])
    # test_set = datasets.OASISBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    # file_names = glob.glob(test_dir + '/*.pkl')
    with torch.no_grad():
        stdy_idx = 0
        for data in val_loader:
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            # x, y, x_seg, y_seg = utils.pkload(data)
            # x, y = x[None, None, ...], y[None, None, ...]
            # x = np.ascontiguousarray(x)
            # y = np.ascontiguousarray(y)
            # x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
            # file_name = file_names[stdy_idx].split('\\')[-1].split('.')[0][2:]
            # print(file_name)
            model.eval()
            x_in = torch.cat((x, y),dim=1)
            x_def, flow = model(x_in)
            x_def = x_def.detach().cpu().numpy()
            x_def = np.squeeze(x_def)

            x = np.squeeze(x.detach().cpu().numpy())
            y = np.squeeze(y.detach().cpu().numpy())


            sitk.WriteImage(sitk.GetImageFromArray(x_def),
                            os.path.join(save_dir,'x_def_{}.nii'.format(stdy_idx)))
            sitk.WriteImage(sitk.GetImageFromArray(x),
                            os.path.join(save_dir,'x_{}.nii'.format(stdy_idx)))
            sitk.WriteImage(sitk.GetImageFromArray(y),
                            os.path.join(save_dir,'y_{}.nii'.format(stdy_idx)))
            flow = flow.cpu().detach().numpy()[0]
            flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
            print(flow.shape)
            # np.savez(save_dir+'/disp_{}.npz'.format(file_name), flow)
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