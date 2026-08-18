import json
import unittest

import numpy as np
from matplotlib import pyplot as plt


def process_loss(loss_file):
    print(f"file path is {loss_file}")
    # 读取 JSON 文件
    with open(loss_file, "r") as file:
        loss_dict = json.load(file)
    ori_loss_ncc = []
    method_loss_ncc = []
    ori_loss_ssim = []
    method_loss_ssim = []
    ori_loss_mi = []
    method_loss_mi = []
    ori_loss_mse = []
    method_loss_mse = []
    # 打印数据示例
    for study_idx, losses in loss_dict.items():
        # print(f"Study Index: {study_idx}")
        # print(f"  NCC Loss: {losses['ncc']}")
        # print(f"  SSIM Loss: {losses['ssim']}")
        # print(f"  MI Loss: {losses['mi']}")
        # print(f"  MSE Loss: {losses['mse']}")

        ori_loss_ncc.append(losses['ncc'][1])
        ori_loss_ssim.append(losses['ssim'][1])
        ori_loss_mi.append(losses['mi'][1])
        ori_loss_mse.append(losses['mse'][1])

        method_loss_ncc.append(losses['ncc'][0])
        method_loss_ssim.append(losses['ssim'][0])
        method_loss_mi.append(losses['mi'][0])
        method_loss_mse.append(losses['mse'][0])

    ori_mean_list = [np.mean(ori_loss_ncc),np.mean(ori_loss_ssim),np.mean(ori_loss_mi),np.mean(ori_loss_mse) * 1e3]
    method_mean_list = [np.mean(method_loss_ncc),np.mean(method_loss_ssim),np.mean(method_loss_mi),np.mean(method_loss_mse) * 1e3]
    # print(f"ori_mean_list is {ori_mean_list}")
    # print(f"method_mean_list is {method_mean_list}")

    # return [method_loss_ncc,method_loss_ssim,method_loss_mi,method_loss_mse],\
    #         [ori_loss_ncc,ori_loss_ssim,ori_loss_mi,ori_loss_mse]
    return method_mean_list, ori_mean_list
def process_loss_avg(loss_file):
    print(f"file path is {loss_file}")
    # 读取 JSON 文件
    with open(loss_file, "r") as file:
        loss_dict = json.load(file)
    ori_loss_ncc = []
    method_loss_ncc = []
    ori_loss_ssim = []
    method_loss_ssim = []
    ori_loss_mi = []
    method_loss_mi = []
    ori_loss_mse = []
    method_loss_mse = []
    ori_loss_dice = []
    method_loss_dice = []
    # 打印数据示例
    for study_idx, losses in loss_dict.items():
        # print(f"Study Index: {study_idx}")
        # print(f"  NCC Loss: {losses['ncc']}")
        # print(f"  SSIM Loss: {losses['ssim']}")
        # print(f"  MI Loss: {losses['mi']}")
        # print(f"  MSE Loss: {losses['mse']}")

        ori_loss_ncc.append(losses['ncc'][1])
        ori_loss_ssim.append(losses['ssim'][1])
        ori_loss_mi.append(losses['mi'][1])
        ori_loss_mse.append(losses['mse'][1])
        ori_loss_dice.append(losses['dice'][1])

        method_loss_ncc.append(losses['ncc'][0])
        method_loss_ssim.append(losses['ssim'][0])
        method_loss_mi.append(losses['mi'][0])
        method_loss_mse.append(losses['mse'][0])
        method_loss_dice.append(losses['dice'][0])

    ori_mean_list = [np.mean(ori_loss_ncc),np.mean(ori_loss_ssim),np.mean(ori_loss_mi),np.mean(ori_loss_mse) * 1e3]
    method_mean_list = [np.mean(method_loss_ncc),np.mean(method_loss_ssim),np.mean(method_loss_mi),np.mean(method_loss_mse) * 1e3]
    print(f"ori_mean_list is {ori_mean_list}")
    print(f"method_mean_list is {method_mean_list}")

    return [np.mean(method_loss_ncc),np.mean(method_loss_ssim)
        ,np.mean(method_loss_mi),np.mean(method_loss_mse), np.mean(method_loss_dice)]

class TestLossND(unittest.TestCase):
    def setUp(self):
        print("start")

    def test_oasis(self):
        # file_path = r"D:\USERS\yq\code\TransMorph_Transformer\OASIS\TransMorph\vxm_oasis_noDsc_0224_loss.json"
        file_path = r"D:\USERS\yq\code\TransMorph_Transformer\OASIS\TransMorph\vxm_oasis_noDsc_0313_loss.json"
        vxm_lists, ori_lists = process_loss(file_path)
        print(f"vxm_lists is {vxm_lists}")
        print(f"ori_lists is {ori_lists}")

        # file_path = r"D:\USERS\yq\code\TransMorph_Transformer\OASIS\TransMorph\trans_oasis_noDsc_0228_loss.json"
        file_path = r"D:\USERS\yq\code\TransMorph_Transformer\OASIS\TransMorph\trans_oasis_noDsc_0313_loss.json"
        trans_lists, ori_lists = process_loss(file_path)
        print(f"trans_lists is {trans_lists}")

        file_path = r"D:\USERS\yq\code\TransMorph_Transformer\TransMorph_MultiScale\dust_oasis_noDsc_0313_loss.json"
        dust_lists, ori_lists = process_loss(file_path)
        print(f"dust_lists is {dust_lists}")



    def test_read_sim(self):
        file_path = r"D:\USERS\yq\code\TransMorph_Transformer\TransMorph\yqScript\Train_0210_Semi\trans_0221_sim_loss.json"
        trans_simple_lists = process_loss_avg(file_path)
        print(f"trans_simple_lists: {trans_simple_lists}")


    def test_Read(self):
        import json

        # 定义文件路径
        file_path = r"D:\USERS\yq\code\TransMorph_Transformer\TransMorph\yqScript\Train1114\trans_simple_loss.json"
        trans_simple_lists, ori_lists = process_loss(file_path)

        file_path = r"D:\USERS\yq\code\TransMorph_Transformer\TransMorph_MultiScale\double_trans_simple_loss.json"
        double_trans_simple_lists, _ = process_loss(file_path)

        file_path = r"D:\USERS\yq\code\TransMorph_Transformer\VoxelMorph_YQ\vxm_loss.json"
        vxm_lists,_ = process_loss(file_path)
        loss_names = ["NCC Loss", "SSIM Loss", "MI Loss", "MSE Loss"]
        num_losses = len(loss_names)
        plt.figure(figsize=(15, 8))

        for i in range(num_losses):
            plt.subplot(2, 2, i + 1)
            data = [
                trans_simple_lists[i],
                double_trans_simple_lists[i],
                vxm_lists[i],
                ori_lists[i]
            ]
            plt.boxplot(data, notch=True, patch_artist=True, showmeans=True,
                        labels=['TransMorph Simple', 'Double TransMorph Simple', 'VXM', 'Original'])
            plt.title(f"Boxplot for {loss_names[i]}")
            plt.ylabel("Accuracy Value")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()





